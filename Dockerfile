FROM python:3.10-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory
RUN mkdir -p models

# Copy the rest of the application
COPY . .

# Ensure models directory has correct permissions
RUN chmod -R 777 models

# Cloud Run sets PORT environment variable to 8080
# Don't set a default PORT - let Cloud Run provide it
EXPOSE 8080

# Command to run the application
# Use $PORT from Cloud Run environment, app is in app/main.py
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT