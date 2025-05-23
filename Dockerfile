# Use a Python base image with a supported version
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY . .

# Install required Python packages
RUN pip install torch flask waitress pillow ultralytics

# Run the Flask app
CMD ["python", "craters_detector.py"]
