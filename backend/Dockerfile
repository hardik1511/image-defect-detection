# syntax=docker/dockerfile:1

# Use a slim Python 3.10 base to match runtime.txt
FROM python:3.10-slim AS base

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system deps for scientific stack and OpenCV pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement files first for better caching
COPY backend/requirements.txt ./

# Install python deps (CPU-only PyTorch per requirements.txt)
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy application code and model weights
COPY backend/ .

# Expose port for Uvicorn
EXPOSE 8000

# Use Uvicorn to run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]



