# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for TA-Lib or other C-based ML libraries if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY src/ ./src/
COPY config/ ./config/
# Ensure datasets folder exists for local file operations
RUN mkdir -p datasets 

# Set the default command (can be overridden by ACI --command-line)
CMD ["python", "src/orchestrator.py", "--symbol", "BTCUSDT", "--mode", "verify"]