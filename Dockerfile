# FinReport RAG - Streamlit Application
# Multi-stage build for a lean production image
FROM python:3.14-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/finrag

# ---- Builder stage ----
FROM base AS builder

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer-cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Runtime stage ----
FROM base AS runtime

WORKDIR /app

# Install only runtime system dependencies (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages and binaries from builder
COPY --from=builder /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create dedicated non-root user with home directory
RUN groupadd -r finrag && useradd -r -g finrag -m -d /home/finrag finrag

# Copy application source (respects .dockerignore)
COPY . .

# Create necessary runtime directories and ensure writable permissions
RUN mkdir -p backend/app/db/chroma_db \
             sample_data \
             logs \
             temp_uploads \
             /home/finrag/.streamlit \
    && chown -R finrag:finrag /app /home/finrag \
    && chmod -R 775 /app \
    && chmod -R 777 /app/backend/app/db/chroma_db /app/logs /app/temp_uploads

USER finrag

# Expose Streamlit port
EXPOSE 8501

# Streamlit environment variables
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the application (CMD allows overrides for debugging or shell access)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
