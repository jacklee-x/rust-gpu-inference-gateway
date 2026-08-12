#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

namespace {

std::string trim(const std::string& value) {
    const std::string whitespace = " \t\r\n";
    const auto begin = value.find_first_not_of(whitespace);
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = value.find_last_not_of(whitespace);
    return value.substr(begin, end - begin + 1);
}

std::string json_escape(const std::string& value) {
    std::string output;
    output.reserve(value.size() + 8);
    for (char ch : value) {
        switch (ch) {
            case '\\': output += "\\\\"; break;
            case '"': output += "\\\""; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default: output += ch; break;
        }
    }
    return output;
}

std::string find_json_string_value(const std::string& json, const std::string& key) {
    const std::string key_pattern = "\"" + key + "\":\"";
    const auto pos = json.find(key_pattern);
    if (pos == std::string::npos) {
        return "";
    }

    const auto start = pos + key_pattern.size();
    const auto end = json.find('"', start);
    if (end == std::string::npos) {
        return "";
    }

    return json.substr(start, end - start);
}

int find_json_int_value(const std::string& json, const std::string& key, int fallback = 0) {
    const std::string key_pattern = "\"" + key + "\":";
    const auto pos = json.find(key_pattern);
    if (pos == std::string::npos) {
        return fallback;
    }

    const auto start = pos + key_pattern.size();
    const auto end = json.find_first_not_of("0123456789-", start);
    const std::string digits = json.substr(start, (end == std::string::npos ? json.size() : end) - start);
    if (digits.empty()) {
        return fallback;
    }
    return std::stoi(digits);
}

float find_json_float_value(const std::string& json, const std::string& key, float fallback = 0.0f) {
    const std::string key_pattern = "\"" + key + "\":";
    const auto pos = json.find(key_pattern);
    if (pos == std::string::npos) {
        return fallback;
    }

    const auto start = pos + key_pattern.size();
    const auto end = json.find_first_not_of("0123456789.-+eE", start);
    const std::string digits = json.substr(start, (end == std::string::npos ? json.size() : end) - start);
    if (digits.empty()) {
        return fallback;
    }
    return std::stof(digits);
}

std::string extract_value_from_nested_options(const std::string& json, const std::string& key, const std::string& fallback = "") {
    const std::string options_key = "\"options\"";
    const auto options_pos = json.find(options_key);
    if (options_pos == std::string::npos) {
        return fallback;
    }

    const auto value_start = json.find('"', options_pos + options_key.size());
    if (value_start == std::string::npos) {
        return fallback;
    }

    const auto candidate = json.substr(value_start + 1);
    const std::string key_pattern = "\"" + key + "\":\"";
    const auto pos = candidate.find(key_pattern);
    if (pos == std::string::npos) {
        return fallback;
    }
    const auto begin = pos + key_pattern.size();
    const auto end = candidate.find('"', begin);
    if (end == std::string::npos) {
        return fallback;
    }
    return candidate.substr(begin, end - begin);
}

struct InferenceRequest {
    std::string model = "llama-7b";
    std::string input = "hello world";
    int max_tokens = 32;
    float temperature = 0.8f;
    float top_p = 0.95f;
};

InferenceRequest parse_request_body(const std::string& body) {
    InferenceRequest request;
    request.model = find_json_string_value(body, "model");
    if (request.model.empty()) {
        request.model = "llama-7b";
    }

    request.input = find_json_string_value(body, "input");
    if (request.input.empty()) {
        request.input = "hello world";
    }

    request.max_tokens = find_json_int_value(body, "max_tokens", 32);
    request.temperature = find_json_float_value(body, "temperature", 0.8f);
    request.top_p = find_json_float_value(body, "top_p", 0.95f);

    if (request.max_tokens <= 0) {
        request.max_tokens = 32;
    }
    if (request.temperature <= 0.0f) {
        request.temperature = 0.8f;
    }
    if (request.top_p <= 0.0f) {
        request.top_p = 0.95f;
    }

    const std::string nested = extract_value_from_nested_options(body, "max_tokens");
    if (!nested.empty()) {
        request.max_tokens = std::stoi(nested);
    }
    return request;
}

std::string make_request_id() {
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    std::ostringstream oss;
    oss << "cpp-" << now_ms;
    return oss.str();
}

std::string run_cpu_inference(const InferenceRequest& request) {
    std::string text = request.input;
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    });

    std::ostringstream oss;
    oss << "C++ CPU inference result for model=" << request.model
        << " input=" << json_escape(request.input)
        << " tokens=" << request.max_tokens
        << " temperature=" << request.temperature
        << " top_p=" << request.top_p
        << " normalized=" << json_escape(text);
    return oss.str();
}

#if defined(__CUDACC__)
std::string run_cuda_inference(const InferenceRequest& request) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count <= 0) {
        return run_cpu_inference(request);
    }

    std::string input_copy = request.input;
    std::vector<char> bytes(input_copy.begin(), input_copy.end());
    bytes.push_back('\0');

    char* device_buffer = nullptr;
    cudaMalloc((void**)&device_buffer, bytes.size());
    cudaMemcpy(device_buffer, bytes.data(), bytes.size(), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (static_cast<int>(bytes.size()) + block_size - 1) / block_size;
    cuda_kernel<<<grid_size, block_size>>>(device_buffer, static_cast<int>(bytes.size()));
    cudaDeviceSynchronize();

    char host_buffer[1024] = {0};
    cudaMemcpy(host_buffer, device_buffer, sizeof(host_buffer), cudaMemcpyDeviceToHost);
    cudaFree(device_buffer);

    std::ostringstream oss;
    oss << "CUDA inference executed on GPU for model=" << request.model
        << " input=" << json_escape(request.input)
        << " output=" << json_escape(host_buffer);
    return oss.str();
}

__global__ void cuda_kernel(char* buffer, int length) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        buffer[idx] = static_cast<char>(buffer[idx] + 1);
    }
}
#else
std::string run_cuda_inference(const InferenceRequest& request) {
    return run_cpu_inference(request);
}
#endif

std::string build_inference_json(const InferenceRequest& request) {
    const std::string result = run_cuda_inference(request);
    std::ostringstream json;
    json << "{"
         << "\"request_id\":\"" << make_request_id() << "\","
         << "\"model\":\"" << json_escape(request.model) << "\","
         << "\"output\":\"" << json_escape(result) << "\","
         << "\"usage\":{\"latency_ms\":" << 5 << ",\"tokens\":" << request.max_tokens << "},"
         << "\"status\":\"ok\""
         << "}";
    return json.str();
}

std::string build_error_json(const std::string& message, int status_code) {
    std::ostringstream json;
    json << "{\"status\":\"error\",\"message\":\"" << json_escape(message) << "\",\"code\":" << status_code << "}";
    return json.str();
}

bool read_available_data(int socket_fd, std::string& output, size_t expected_bytes) {
    output.clear();
    char buffer[4096];
    size_t received_total = 0;
    while (received_total < expected_bytes) {
        const int chunk_size = static_cast<int>(std::min<size_t>(sizeof(buffer), expected_bytes - received_total));
#if defined(_WIN32)
        const int received = recv(socket_fd, buffer, chunk_size, 0);
#else
        const ssize_t received = recv(socket_fd, buffer, chunk_size, 0);
#endif
        if (received <= 0) {
            return false;
        }
        output.append(buffer, static_cast<size_t>(received));
        received_total += static_cast<size_t>(received);
    }
    return true;
}

std::string read_http_request(int socket_fd) {
    std::string request;
    char buffer[4096];
    while (true) {
#if defined(_WIN32)
        const int received = recv(socket_fd, buffer, sizeof(buffer), 0);
#else
        const ssize_t received = recv(socket_fd, buffer, sizeof(buffer), 0);
#endif
        if (received <= 0) {
            return request;
        }
        request.append(buffer, static_cast<size_t>(received));
        if (request.find("\r\n\r\n") != std::string::npos) {
            break;
        }
        if (request.size() > 64 * 1024) {
            break;
        }
    }

    const std::string header = request.substr(0, request.find("\r\n\r\n"));
    const std::string content_length_header = "Content-Length:";
    const auto position = header.find(content_length_header);
    if (position != std::string::npos) {
        const auto value_start = position + content_length_header.size();
        const auto value_end = header.find("\r\n", value_start);
        const std::string length_value = trim(header.substr(value_start, value_end - value_start));
        const size_t content_length = static_cast<size_t>(std::stoul(length_value));
        const auto body_start = request.find("\r\n\r\n");
        if (body_start != std::string::npos) {
            const std::string existing_body = request.substr(body_start + 4);
            if (existing_body.size() < content_length) {
                std::string rest;
                if (read_available_data(socket_fd, rest, content_length - existing_body.size())) {
                    request.append(rest);
                }
            }
        }
    }
    return request;
}

void send_response(int client_socket, const std::string& status_line, const std::string& payload) {
    const std::string response =
        status_line + "Content-Type: application/json\r\n"
        + "Content-Length: " + std::to_string(payload.size()) + "\r\n"
        + "Connection: close\r\n\r\n" + payload;
#if defined(_WIN32)
    send(client_socket, response.c_str(), static_cast<int>(response.size()), 0);
#else
    send(client_socket, response.c_str(), response.size(), 0);
#endif
}

void handle_client_connection(int client_socket) {
    const std::string request = read_http_request(client_socket);
    if (request.empty()) {
        send_response(client_socket, "HTTP/1.1 400 Bad Request\r\n", build_error_json("empty request", 400));
#if defined(_WIN32)
        closesocket(client_socket);
#else
        close(client_socket);
#endif
        return;
    }

    const auto request_line_end = request.find("\r\n");
    const std::string request_line = request.substr(0, request_line_end);
    std::istringstream iss(request_line);
    std::string method;
    std::string path;
    std::string version;
    iss >> method >> path >> version;

    if (method == "GET" && path == "/health") {
        const std::string payload = "{\"status\":\"healthy\",\"engine\":\"c++-cuda-ready\"}";
        send_response(client_socket, "HTTP/1.1 200 OK\r\n", payload);
#if defined(_WIN32)
        closesocket(client_socket);
#else
        close(client_socket);
#endif
        return;
    }

    if (method == "POST" && path == "/infer") {
        const auto body_start = request.find("\r\n\r\n");
        if (body_start == std::string::npos) {
            send_response(client_socket, "HTTP/1.1 400 Bad Request\r\n", build_error_json("missing body", 400));
#if defined(_WIN32)
            closesocket(client_socket);
#else
            close(client_socket);
#endif
            return;
        }

        const std::string body = request.substr(body_start + 4);
        const InferenceRequest request_details = parse_request_body(body);
        const std::string payload = build_inference_json(request_details);
        send_response(client_socket, "HTTP/1.1 200 OK\r\n", payload);
#if defined(_WIN32)
        closesocket(client_socket);
#else
        close(client_socket);
#endif
        return;
    }

    send_response(client_socket, "HTTP/1.1 404 Not Found\r\n", build_error_json("route not found", 404));
#if defined(_WIN32)
    closesocket(client_socket);
#else
    close(client_socket);
#endif
}

void serve_forever(uint16_t port) {
#if defined(_WIN32)
    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
        std::cerr << "Failed to initialize Winsock" << std::endl;
        std::exit(1);
    }
#endif

    const int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::cerr << "Failed to create socket" << std::endl;
        std::exit(1);
    }

    int opt = 1;
#if defined(_WIN32)
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt));
#else
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = inet_addr("127.0.0.1");

    if (bind(server_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
        std::cerr << "Bind failed on port " << port << std::endl;
        std::exit(1);
    }

    if (listen(server_fd, 16) < 0) {
        std::cerr << "Listen failed" << std::endl;
        std::exit(1);
    }

    std::cout << "C++ inference core listening on 127.0.0.1:" << port << std::endl;
    while (true) {
        sockaddr_in client_address{};
#if defined(_WIN32)
        int client_length = sizeof(client_address);
        const SOCKET client_fd = accept(server_fd, reinterpret_cast<sockaddr*>(&client_address), &client_length);
#else
        socklen_t client_length = sizeof(client_address);
        const int client_fd = accept(server_fd, reinterpret_cast<sockaddr*>(&client_address), &client_length);
#endif
        if (client_fd < 0) {
            std::cerr << "Accept failed" << std::endl;
            continue;
        }

        std::thread client_thread(handle_client_connection, client_fd);
        client_thread.detach();
    }
}

}  // namespace

int main(int argc, char** argv) {
    int port = 8081;
    if (argc >= 3 && std::string(argv[1]) == "--port") {
        port = std::stoi(argv[2]);
    }

    std::cout << "Starting C++ inference core with optional CUDA path" << std::endl;
    serve_forever(static_cast<uint16_t>(port));
    return 0;
}
