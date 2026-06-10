$ErrorActionPreference = "Stop"

Write-Host "Stopping old container..."
docker compose -f docker-compose.cpu.yml down --remove-orphans
docker rm -f ai-prompt-sim-rag-cpu 2>$null | Out-Null

Write-Host "Removing old image so the CRLF fix is definitely used..."
docker image rm ai-prompt-sim-rag:cpu 2>$null | Out-Null

Write-Host "Building fresh image..."
docker compose -f docker-compose.cpu.yml build --no-cache

Write-Host "Starting container..."
docker compose -f docker-compose.cpu.yml up
