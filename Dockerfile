# ============================================================
# Sepsis Early Prediction System — Production Dockerfile
# ============================================================
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Set workspace directory
WORKDIR /app

# Install system dependencies (libgomp1 is required by LightGBM/XGBoost on Linux)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .

# Install CPU-only PyTorch first (avoids pulling ~3.5 GB CUDA bundles)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (skip torch/torchvision since already installed)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application directories and files
COPY backend/ ./backend/
COPY dashboard/ ./dashboard/
COPY results/ ./results/
COPY src/ ./src/
COPY config.py start_server.py ./

# Expose port
EXPOSE 8000

# Run the server
CMD ["python", "-u", "start_server.py"]
