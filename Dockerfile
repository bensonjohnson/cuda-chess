# Use the official Python image as the base image
FROM python:3.9-slim-buster

# Install CUDA and cuDNN
RUN apt-get update && \
    apt-get install -y --no-install-recommends gnupg && \
    echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" >> /etc/apt/sources.list.d/cuda.list && \
    echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" >> /etc/apt/sources.list.d/cuda.list && \
    apt-key adv --fetch-keys "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub" && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        cuda-cudart-11-0 \
        libcudnn8=8.0.4.30-1+cuda11.0 \
        libcudnn8-dev=8.0.4.30-1+cuda11.0 && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory to /app
WORKDIR /app

# Copy the Python requirements file into the container
COPY requirements.txt .

# Install the Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . .

# Set the environment variables for CUDA and cuDNN
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME /usr/local/cuda

# Expose port 8000 for the Flask app
EXPOSE 8000

# Start the Flask app
CMD ["python", "app.py"]
