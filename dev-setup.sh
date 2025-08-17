#!/usr/bin/env bash

echo "Setting up development environment..."
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi
echo "Creating virtual environment and installing dependencies..."
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
echo "Development environment setup complete!"
echo "To activate: source .venv/bin/activate"
