# Simple one-off Windows runner for GGUF quantization
# Usage: .\run-windows.ps1 [-cuda] [args...]

param(
    [switch]$Cuda,
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Arguments = @()
)

if ($Cuda) {
    $IMAGE_NAME = "gguf-quantization-cuda"
    $GPU_ARGS = @("--gpus", "all")
    $BUILD_FILE = "Dockerfile.cuda"
    $ENV_VARS = @("-e", "CUDA_VISIBLE_DEVICES=$($env:CUDA_VISIBLE_DEVICES ?? '0')")
    $BUILD_ARGS = @("--build-arg", "CUDA_ARCHITECTURES=$(Get-CudaArchitectures)", "--build-arg", "THREADS=$([Environment]::ProcessorCount)")
    $TYPE = "CUDA"
} else {
    $IMAGE_NAME = "gguf-quantization-cpu"
    $GPU_ARGS = @()
    $BUILD_FILE = "Dockerfile.cpu"
    $ENV_VARS = @()
    $BUILD_ARGS = @("--build-arg", "THREADS=$([Environment]::ProcessorCount)")
    $TYPE = "CPU"
}

$CONTAINER_NAME = "gguf-windows"

function Write-ColorOutput([string]$ForegroundColor, [string]$Message) {
    $originalColor = $Host.UI.RawUI.ForegroundColor
    $Host.UI.RawUI.ForegroundColor = $ForegroundColor
    Write-Output $Message
    $Host.UI.RawUI.ForegroundColor = $originalColor
}

function Get-CudaArchitectures {
    try {
        $gpuInfo = nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>$null | Select-Object -First 1
        if ($gpuInfo) {
            $major, $minor = $gpuInfo -split '\.'
            return "$major$minor"
        }
    } catch {}
    return "75;80;86;89;90"
}

# Check if image exists, build if not
try {
    docker image inspect $IMAGE_NAME | Out-Null
} catch {
    Write-ColorOutput "Yellow" "Building GGUF $TYPE image..."
    if ($Cuda) { Write-Output "CUDA architectures: $(Get-CudaArchitectures)" }
    $buildCmd = @("build", "-f", $BUILD_FILE, "-t", $IMAGE_NAME) + $BUILD_ARGS + @(".")
    & docker @buildCmd
    Write-ColorOutput "Green" "$TYPE build completed!"
}

# Create directories and run
Write-ColorOutput "Yellow" "Running GGUF quantization ($TYPE)..."
New-Item -ItemType Directory -Force -Path "models", "output", ".cache" | Out-Null

# Show system info
if ($Cuda) {
    try {
        Write-ColorOutput "Blue" "Available GPUs:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
    } catch {}
} else {
    Write-ColorOutput "Blue" "System info:"
    Write-Output "CPU cores: $([Environment]::ProcessorCount)"
    Write-Output "Available memory: $([math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 2)) GB"
}

$currentDir = (Get-Location).Path
$dockerCmd = @("run", "--rm", "--name", $CONTAINER_NAME) + $GPU_ARGS + @(
    "-v", "${currentDir}\models:/app/GGUF/models"
    "-v", "${currentDir}\output:/app/GGUF/output" 
    "-v", "${currentDir}\.cache:/root/.cache"
    "-e", "HF_TOKEN=$env:HF_TOKEN"
    "-e", "THREADS=$([Environment]::ProcessorCount)"
) + $ENV_VARS + @($IMAGE_NAME, "python3", "pipeline.py") + $Arguments

& docker @dockerCmd 