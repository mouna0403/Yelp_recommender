FROM python:3.10-slim

# ----------------------------
# 1. Install minimal system dependencies
# Required for compiling Python packages when no wheels are available
# ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# 2. Install uv (fast Python package manager)
# ----------------------------
RUN pip install uv

# ----------------------------
# 3. Set working directory
# ----------------------------
WORKDIR /app

# ----------------------------
# 4. Copy dependency files first to leverage Docker cache
# ----------------------------
COPY pyproject.toml uv.lock ./

# ----------------------------
# 5. Install PyTorch separately using official CPU wheels
# This avoids heavy dependency resolution inside sentence-transformers
# ----------------------------
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# ----------------------------
# 6. Install remaining dependencies using uv with pip cache enabled
# ----------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    uv pip install --system .

# ----------------------------
# 7. Copy application source code
# ----------------------------
COPY . .

# ----------------------------
# 8. Expose Streamlit default port
# ----------------------------
EXPOSE 8501

# ----------------------------
# 9. Run Streamlit application
# ----------------------------
CMD ["streamlit", "run", "src/Yelp_recommender/main.py", "--server.port=8501", "--server.address=0.0.0.0"]