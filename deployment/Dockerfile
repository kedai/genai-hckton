# Base image
FROM python:3.10-slim
# FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Set the working directory
WORKDIR /app

# Install system dependencies for image processing and PDF handling
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    libtesseract-dev \
    build-essential \
    wget \
    gnupg2 \
    ca-certificates \
    poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY . .

# Expose the port that Streamlit uses
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "simple-app-dbx.py"]
