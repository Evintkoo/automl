# ============================================================================
# AutoML - Multi-stage Docker Build for Railway
# ============================================================================
# Stage 1: Build the Rust binary with full optimizations
# Stage 2: Minimal runtime image with only what's needed
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder
# ---------------------------------------------------------------------------
FROM rust:1.91-bookworm AS builder

WORKDIR /app

# Install build dependencies for native libraries (OpenSSL, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Cache dependencies: copy manifests first, build a dummy project,
# then replace with real source. This avoids re-downloading deps on
# every source change.
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    echo "pub fn lib() {}" > src/lib.rs && \
    cargo build --release 2>/dev/null || true && \
    rm -rf src

# Copy the full source tree
COPY src/ src/

# Touch main.rs so cargo sees the real source as newer than the dummy
RUN touch src/main.rs src/lib.rs

# Build the release binary with full optimizations (LTO, single codegen-unit)
RUN cargo build --release --bin automl

# ---------------------------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

# Install minimal runtime dependencies
# - ca-certificates: for HTTPS/TLS in reqwest
# - libssl3: OpenSSL runtime (linked by reqwest)
# - curl: for Docker HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN groupadd -r automl && useradd -r -g automl -m automl

WORKDIR /app

# Copy the compiled binary from the builder stage
COPY --from=builder /app/target/release/automl /app/automl

# Copy the web UI static assets
COPY automl-web/ /app/automl-web/

# Create runtime directories for data and models
RUN mkdir -p /app/data /app/models && \
    chown -R automl:automl /app

# Switch to non-root user
USER automl

# ---------------------------------------------------------------------------
# Environment Configuration
# ---------------------------------------------------------------------------
ENV API_HOST=0.0.0.0 \
    STATIC_DIR=/app/automl-web/static \
    DATA_DIR=/app/data \
    MODELS_DIR=/app/models \
    MAX_UPLOAD_SIZE=104857600 \
    RUST_LOG=automl=info,tower_http=info

# Railway dynamically assigns the port via $PORT env var.
# EXPOSE is informational; Railway routes traffic to $PORT automatically.
EXPOSE 8080

# Health check — uses $PORT at runtime (Railway sets it; defaults to 8080)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/api/health || exit 1

# Shell form so $PORT is expanded at container start.
# Railway sets PORT; if unset, falls back to 8080.
CMD /app/automl serve --host 0.0.0.0 --port ${PORT:-8080}
