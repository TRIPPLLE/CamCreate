# Use an official PyTorch runtime as a parent image (includes CUDA support)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Install necessary Python packages
# (PyTorch is already installed in the base image)
RUN pip install --no-cache-dir fastapi uvicorn python-multipart Pillow numpy torchvision

# Copy the current directory contents into the container at /app
# Note: The .dockerignore file ensures large datasets and frontend code are skipped
COPY . /app/

# Expose the port the app runs on (4567)
EXPOSE 4567

# Run the FastAPI server
CMD ["python", "api.py"]
