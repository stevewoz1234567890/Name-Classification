# Use the official Python image from the Docker Hub
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files to the working directory
COPY . .

# Expose the port that the Flask app will run on
EXPOSE 5000

# Set the environment variable to disable buffering in the Flask app output (for logging)
ENV PYTHONUNBUFFERED=1

# Command to run the Flask application
CMD ["python", "scripts/app.py"]
