FROM python:3.10-slim

# Ensure Python behaves predictably
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/cache/hf

WORKDIR /app

# System deps often needed for building/using common ML libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better caching
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the app (compose will mount over this for dev)
COPY . .

# Default shell; compose overrides command as needed
CMD ["bash"]