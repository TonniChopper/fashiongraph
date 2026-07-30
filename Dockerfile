# ---- 1. build the front end ----
FROM node:20-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ---- 2. backend + serve ----
FROM python:3.11-slim
WORKDIR /app

# System deps for the vision/OCR stack (kept minimal; comment out if unused).
RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY fg ./fg
RUN pip install --no-cache-dir -e . \
 && pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart

# the compiled front end (served at / by FastAPI)
COPY --from=web /web/dist ./frontend/dist

# Ollama runs on the HOST — point the app at it (Linux: --add-host or 172.17.0.1)
ENV FG_OLLAMA_HOST=http://host.docker.internal:11434
EXPOSE 8000
CMD ["uvicorn", "fg.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
