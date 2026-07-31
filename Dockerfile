# Dockerfile for FastAPI Backend
FROM python:3.10-slim

WORKDIR /workspace/backend

# Install system dependencies needed for compiling postgres client and packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from the build context (which is ./backend)
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all backend source code under a backend/ subdirectory
COPY . /workspace/backend

EXPOSE 8000

ENV PYTHONPATH=/workspace/backend

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
