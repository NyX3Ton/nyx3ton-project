docker compose -f docker-compose.cpu.yml down --remove-orphans
docker rm -f ai-prompt-sim-rag-cpu 2>$null
