# start-llama-server.ps1
# ------------------------
# Windows helper to run llama.cpp llama-server (the real GPU inference
# core) for local development, then point the Rust gateway at it with
# CORE_URL and CORE_PROTOCOL=llama-chat.
#
# Two serving modes:
#   - Single model: pass -Model (a GGUF file). llama-server runs as the
#     classic single-model instance.
#   - Multi model: pass -ModelsDir (a directory of GGUF files).
#     llama-server runs in router mode: every model in the directory is
#     served in its own lazy-loaded child process and appears in
#     GET /v1/models, which the gateway mirrors dynamically.
#
# Prerequisites:
#   - llama.cpp built with CUDA support (GGML_CUDA=ON), llama-server.exe
#     on PATH or passed via -LlamaServer
#   - a GGUF model file (default: models/qwen2.5-0.5b-instruct-q4_k_m.gguf)
#
# Usage:
#   .\scripts\start-llama-server.ps1 [-Port 8081] [-Model ..\models\xxx.gguf]
#     [-ModelsDir ..\models] [-Layers 99] [-Context 2048]
#     [-LlamaServer C:\path\to\llama-server.exe]

param(
    [int]$Port = 8081,
    [string]$Model = "..\models\qwen2.5-0.5b-instruct-q4_k_m.gguf",
    [string]$ModelsDir = "",
    [int]$Layers = 99,          # layers offloaded to GPU (-ngl)
    [int]$Context = 2048,       # context window size (-c)
    [string]$LlamaServer = "llama-server"
)

$ErrorActionPreference = "Stop"

# llama-server (with the CUDA backend) dynamically links against cuBLAS
# shipped with the CUDA Toolkit. The 64-bit runtime DLLs live in
# <CUDA>/bin/x64 (CUDA 13 layout), which is not on PATH by default — so
# prepend it here, otherwise the process exits with STATUS_DLL_NOT_FOUND.
$cudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64"
if (Test-Path -LiteralPath $cudaBin) {
    $env:PATH = "$cudaBin;$env:PATH"
}

# Resolve the model path for single-model mode. An absolute path is
# used as-is; a relative path is resolved against the script location so
# this script works no matter where it is invoked from.
if ([System.IO.Path]::IsPathRooted($Model)) {
    $ModelPath = $Model
} else {
    $ModelPath = Join-Path $PSScriptRoot $Model
}

$logDir = Join-Path $PSScriptRoot "..\.dev"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$logFile = Join-Path $logDir "llama-server.log"

# Build the llama-server arguments. All GGUF layers are offloaded to the
# GPU (--gpu-layers). -ModelsDir selects router (multi-model) mode:
# every GGUF in the directory is served lazily and listed by
# GET /v1/models. Otherwise a single model is loaded with the alias
# matching the model registry entry.
if ($ModelsDir) {
    if ([System.IO.Path]::IsPathRooted($ModelsDir)) {
        $ModelsDirPath = $ModelsDir
    } else {
        $ModelsDirPath = Join-Path $PSScriptRoot $ModelsDir
    }
    if (-not (Test-Path -LiteralPath $ModelsDirPath)) {
        Write-Error "Models directory not found: $ModelsDirPath"
    }
    $args = @(
        "--host", "127.0.0.1",
        "--port", "$Port",
        "--models-dir", $ModelsDirPath,
        "--gpu-layers", "$Layers",
        "--ctx-size", "$Context"
    )
} else {
    if (-not (Test-Path -LiteralPath $ModelPath)) {
        Write-Error "Model file not found: $ModelPath"
    }
    $args = @(
        "--model", $ModelPath,
        "--host", "127.0.0.1",
        "--port", "$Port",
        "--gpu-layers", "$Layers",
        "--ctx-size", "$Context",
        "--alias", "qwen2.5-0.5b-instruct"
    )
}

Write-Host "Starting llama-server on 127.0.0.1:$Port (log: $logFile)"
$process = Start-Process -FilePath $LlamaServer `
    -ArgumentList $args `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError "$logFile.err" `
    -WindowStyle Hidden -PassThru

# Wait until the server responds on /health before returning, so callers
# (e.g. the gateway) can rely on the core being ready.
$healthUrl = "http://127.0.0.1:$Port/health"
$deadline = (Get-Date).AddSeconds(120)
while ((Get-Date) -lt $deadline) {
    try {
        $response = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            Write-Host "llama-server healthy at $healthUrl (PID: $($process.Id))"
            exit 0
        }
    } catch {
        Start-Sleep -Seconds 3
    }
}

Write-Error "llama-server did not become healthy within 120s. See $logFile"