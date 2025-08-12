ARG BASE_IMAGE_NAME
FROM ${BASE_IMAGE_NAME} AS test

ARG VERSION
ENV VERSION=${VERSION}

ENV PYTHONPATH=/app

WORKDIR /app

# Copy pyproject.toml first for better caching
COPY pyproject.toml .

# Install dependencies including test dependencies
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install -e ".[test,dev]"

# Copy application code
COPY service/ ./service/
COPY tests/ ./tests/
COPY scripts/ ./scripts/

# Set environment to use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy version file if it exists
RUN if [ -n "${VERSION}" ]; then echo "${VERSION}" > service/_version; fi

# Create test reports directory
RUN mkdir -p /tmp/test-reports

# Default command for testing
CMD ["scripts/run_test.sh", "--test-suite", "unit"]