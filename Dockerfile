FROM python:3.11-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-editable

COPY main.py ./
COPY config.py ./
COPY frontend/build ./frontend/build

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
