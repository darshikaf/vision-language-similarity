ARG RAY_IMAGE=rayproject/ray:2.39.0-py310-cpu-aarch64
FROM ${RAY_IMAGE}

# Set user to root for package installation
USER root

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/home/ray/.local/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv using the official installer
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    uv --version

# Create app directory
WORKDIR /app

# Copy pyproject.toml and install dependencies with uv
COPY pyproject.toml /app/

# Install dependencies using uv
RUN uv sync --no-dev

# Create cache directories
RUN mkdir -p /root/.cache/openclip
RUN mkdir -p /home/ray/.cache/openclip

# Set environment to use the virtual environment  
ENV PATH="/app/.venv/bin:$PATH"
