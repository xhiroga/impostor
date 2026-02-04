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
COPY inference.py ./
COPY frontend/build ./frontend/build

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uv", "run", "fastapi", "run", "main.py"]
