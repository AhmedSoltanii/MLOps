# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies for FastAPI and Uvicorn
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install the required Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the web service (main.py contains FastAPI app)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


