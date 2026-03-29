FROM mcr.microsoft.com/devcontainers/python:3.13-bookworm

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock* .python-version /tmp/project/

# Install dependencies (cached layer — only re-runs when deps change)
WORKDIR /tmp/project
RUN uv sync --no-install-project 2>/dev/null || true

WORKDIR /workspaces/eval-workshop-ext
