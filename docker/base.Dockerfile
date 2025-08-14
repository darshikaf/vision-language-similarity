FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv in a separate step to ensure it works properly
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv --version

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

# Create app directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml .

# Install dependencies directly from pyproject.toml
RUN uv sync --no-dev

# Create cache directories
RUN mkdir -p /root/.cache/openclip
RUN mkdir -p /root/.cache/uv

# Set environment to use the virtual environment  
ENV PATH="/app/.venv/bin:$PATH"