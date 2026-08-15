# start-llama-server.ps1
# ------------------------
# Windows helper to run llama.cpp llama-server (the real GPU inference
# core) for local development, then point the Rust gateway at it with
# CORE_URL and CORE_PROTOCOL=llama-chat.
#
# Prerequisites:
#   - llama.cpp built with CUDA support (GGML_CUDA=ON), llama-server.exe
#     on PATH or passed via -LlamaServer
#   - a GGUF model file (default: models/qwen2.5-0.5b-instruct-q4_k_m.gguf)
#
# Usage:
#   .\scripts\start-llama-server.ps1 [-Port 8081] [-Model ..\models\xxx.gguf]
#     [-Layers 99] [-Context 2048] [-LlamaServer C:\path\to\llama-server.exe]

param(
    [int]$Port = 8081,
    [string]$Model = "..\models\qwen2.5-0.5b-instruct-q4_k_m.gguf",
    [int]$Layers = 99,          # layers offloaded to GPU (-ngl)
    [int]$Context = 2048,       # context window size (-c)
    [string]$LlamaServer = "llama-server"
)

$ErrorActionPreference = "Stop"

# Resolve the model path relative to the script location so this script
# works no matter where it is invoked from.
$ModelPath = Join-Path $PSScriptRoot $Model
if (-not (Test-Path -LiteralPath $ModelPath)) {
    Write-Error "Model file not found: $ModelPath"
}

$logDir = Join-Path $PSScriptRoot "..\.dev"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$logFile = Join-Path $logDir "llama-server.log"

# All GGUF layers are offloaded to the GPU (--gpu-layers), so inference
# runs on the NVIDIA device; the alias matches the gateway model registry
# entry so /v1/chat/completions requests are served under that name.
$args = @(
    "--model", $ModelPath,
    "--host", "127.0.0.1",
    "--port", "$Port",
    "--gpu-layers", "$Layers",
    "--ctx-size", "$Context",
    "--alias", "qwen2.5-0.5b-instruct"
)

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