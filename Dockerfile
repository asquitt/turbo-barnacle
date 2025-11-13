# Multi-stage Dockerfile for Autonomous Materials Discovery
# Optimized for size and build speed

# Stage 1: Base image with Python and system dependencies
FROM python:3.9-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies
FROM base as dependencies

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 3: Application
FROM dependencies as application

# Copy application code
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs

# Set Python path
ENV PYTHONPATH=/app

# Default command (can be overridden)
CMD ["python", "src/main.py", "--help"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import pymatgen" || exit 1

# Metadata
LABEL maintainer="Materials Discovery Team"
LABEL description="Autonomous Materials Discovery with LLM-Guided Search"
LABEL version="1.0.0"
