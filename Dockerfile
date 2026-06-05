# Use an official, optimized lightweight Python runtime environment
FROM python:3.10-slim

# Establish the internal container workspace directory path
WORKDIR /app

# Install basic operating system utilities required for compiling dependency packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy your python library requirements sheet first to optimize caching layers
COPY requirements.txt .

# Install all data science and web hosting frameworks
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your deep learning modules, saved weights, and web server scripts into the image
COPY src/ ./src/
COPY best_lstm_weights.pth .
COPY app.py .

# Expose the identical communications networking port designated for FastAPI
EXPOSE 8000

# Launch the production ASGI web server to handle requests 24/7
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]