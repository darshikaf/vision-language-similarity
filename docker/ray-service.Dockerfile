ARG BASE_IMAGE_NAME
FROM ${BASE_IMAGE_NAME} AS runtime

ARG VERSION
ENV VERSION=${VERSION}

ENV PYTHONPATH=/app
# Ensure virtual environment is in PATH
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copy service code
COPY service/ ./service/

# Copy version file if it exists
RUN if [ -n "${VERSION}" ]; then echo "${VERSION}" > service/_version; fi

# Verify Ray installation and deployment
RUN . .venv/bin/activate && \
    ray --version && \
    python -c "import ray; from ray import serve; print('Ray Serve available')" && \
    python -c "from service.main import deployment; print('Deployment created successfully')"

# Expose ports
EXPOSE 8000 8265

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/evaluator/health || exit 1

CMD ["/bin/bash", "-c", "source .venv/bin/activate && serve run service.main:deployment"]
