# Use Python 3.9 as the base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Install dependencies for ffmpeg and clean up the cache afterward
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libavcodec-extra \
    x264 \
    libx264-dev && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y libhdf5-dev

# Copy the requirements file and install dependencies
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app

# Expose the port on which the FastAPI app will run
EXPOSE 8002

# Command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
