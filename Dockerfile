# Alpaca Improved - Production Dockerfile
# Multi-stage build for optimal image size and security

# ============================================================================
# Build Stage - Install dependencies and build the application
# ============================================================================
FROM python:3.9-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.5.1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==$POETRY_VERSION

# Create app directory
WORKDIR /app

# Copy Poetry configuration
COPY pyproject.toml poetry.lock ./

# Configure Poetry
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --only=main --no-dev

# Copy source code
COPY src/ ./src/
COPY README.md ./
COPY LICENSE ./

# Build the package
RUN poetry build

# ============================================================================
# Production Stage - Minimal runtime image
# ============================================================================
FROM python:3.9-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    APP_ENV=production

# Install system runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create app directory and set ownership
WORKDIR /app
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Create necessary directories
RUN mkdir -p /app/{logs,data,config}

# Copy built application from builder stage
COPY --from=builder --chown=appuser:appuser /app/dist/ ./dist/

# Install the built package
USER root
RUN pip install --no-cache-dir ./dist/*.whl
USER appuser

# Copy configuration files
COPY --chown=appuser:appuser config/ ./config/

# Create volume mounts for persistent data
VOLUME ["/app/data", "/app/logs", "/app/config"]

# Expose port for web interface
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "alpaca_improved"]

# ============================================================================
# Development Stage - Full development environment
# ============================================================================
FROM python:3.9-slim as development

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=development \
    POETRY_VERSION=1.5.1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==$POETRY_VERSION

# Create app directory
WORKDIR /app

# Copy Poetry configuration
COPY pyproject.toml poetry.lock ./

# Configure Poetry
RUN poetry config virtualenvs.create false

# Install all dependencies (including dev)
RUN poetry install

# Copy source code
COPY . ./

# Create non-root user for development
RUN groupadd -r devuser && useradd -r -g devuser devuser
RUN chown -R devuser:devuser /app

# Switch to development user
USER devuser

# Expose development ports
EXPOSE 8000 8501 5432 6379

# Development command
CMD ["python", "-m", "alpaca_improved", "--dev"]

# ============================================================================
# Testing Stage - Environment for running tests
# ============================================================================
FROM development as testing

# Switch back to root for installing test dependencies
USER root

# Install additional test dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Switch back to dev user
USER devuser

# Set test environment
ENV APP_ENV=testing

# Run tests by default
CMD ["pytest", "-v", "--cov=src/alpaca_improved"] 