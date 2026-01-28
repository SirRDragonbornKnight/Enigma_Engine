# Forge_AI Docker Configuration
#
# Build: docker build -t forge-ai .
# Run:   docker run --gpus all -p 8000:8000 forge-ai
#
# Multi-stage build for smaller image size

# ============================================
# Stage 1: Build dependencies
# ============================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ============================================
# Stage 2: CUDA Runtime (for GPU support)
# ============================================
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 as cuda-base

# Install Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# ============================================
# Stage 3: Final image
# ============================================
FROM cuda-base as final

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY forge_ai/ ./forge_ai/
COPY run.py .
COPY setup.py .
COPY pyproject.toml .
COPY README.md .

# Install the package
RUN pip install -e .

# Create directories for data persistence
RUN mkdir -p /app/models /app/data /app/outputs /app/logs

# Environment variables
ENV FORGE_MODEL_PATH=/app/models
ENV FORGE_DATA_PATH=/app/data
ENV FORGE_OUTPUT_PATH=/app/outputs
ENV FORGE_LOG_PATH=/app/logs

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: start API server
CMD ["python", "run.py", "--serve", "--api-type", "openai", "--host", "0.0.0.0", "--port", "8000"]

# ============================================
# Alternative: CPU-only image
# ============================================
FROM python:3.11-slim as cpu-only

WORKDIR /app

# Install minimal dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY forge_ai/ ./forge_ai/
COPY run.py .
COPY setup.py .
COPY pyproject.toml .
COPY README.md .

RUN pip install -e .

RUN mkdir -p /app/models /app/data /app/outputs /app/logs

ENV FORGE_MODEL_PATH=/app/models
ENV FORGE_DATA_PATH=/app/data
ENV FORGE_OUTPUT_PATH=/app/outputs
ENV FORGE_LOG_PATH=/app/logs
ENV CUDA_VISIBLE_DEVICES=""

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "run.py", "--serve", "--api-type", "openai", "--host", "0.0.0.0", "--port", "8000"]
