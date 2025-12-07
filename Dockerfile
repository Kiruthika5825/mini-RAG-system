# ---------- Stage 1: Builder ----------
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies (only if you need to compile packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv .venv

# Upgrade pip inside the venv
RUN .venv/bin/pip install --upgrade pip

# Copy requirements and install dependencies into venv
COPY requirements.txt .
RUN .venv/bin/pip install --no-cache-dir -r requirements.txt

# ---------- Stage 2: Runtime ----------
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY . .

# Ensure venv is used by default
ENV PATH="/app/.venv/bin:$PATH"

# Expose FastAPI and Streamlit ports
EXPOSE 8000 8501

# Run both FastAPI and Streamlit
CMD ["bash", "-c", "uvicorn app.api:app --host 0.0.0.0 --port 8000 & streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501"]
