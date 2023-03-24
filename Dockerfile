# Use the official PyTorch image as the base image
FROM pytorch/pytorch:1.10-cuda11.1-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the entry point for the container
CMD ["python", "chess_app.py"]
