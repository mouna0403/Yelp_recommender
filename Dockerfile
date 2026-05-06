FROM python:3.10-slim

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy only dependency files first (cache Docker)
COPY pyproject.toml ./

# Install dependencies from pyproject.toml
RUN uv pip install --system .

# Copy the rest of the project
COPY . .

# Preload transformer model (optional but recommended)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8501

CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]