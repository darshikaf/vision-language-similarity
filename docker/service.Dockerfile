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

# Expose port
EXPOSE 8000

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]