# Use a base image with Python
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose necessary ports
EXPOSE 5000 8000 8501 8080

# Command to run the application
CMD ["python", "scripts/start_complete_deployment.py"]
