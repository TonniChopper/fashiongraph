#!/usr/bin/env bash
# One-command run: build the front end, then serve the whole app (API + UI) on
# http://localhost:8000. Needs Ollama running (`ollama serve`) with the models
# pulled: `ollama pull qwen2.5:7b-instruct qwen2.5vl:7b`.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "→ building the front end…"
( cd frontend && npm install --silent && npm run build )

PORT="${PORT:-8000}"
echo "→ FashionGraph on http://localhost:${PORT}"
exec python -m uvicorn fg.api.app:app --host 0.0.0.0 --port "${PORT}"
