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
RUN mkdir -p models && chmod 777 models

# Copy the rest of the application
COPY . .

# Ensure models directory has correct permissions
RUN chmod -R 777 models

# Set default port if not provided
ENV PORT=8000

# Expose the port
EXPOSE ${PORT}

# Command to run the application
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
