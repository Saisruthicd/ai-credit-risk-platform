FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for ML packages (like lightgbm/xgboost if used)
RUN apt-get update && apt-get install -y libgomp1 gcc python3-dev && rm -rf /var/lib/apt/lists/*

# Copy dependencies first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set python path so local imports like 'src.xxx' resolve
ENV PYTHONPATH=/app
