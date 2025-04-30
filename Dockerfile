# Use a base Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all required files and folders into the container
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port Flask uses
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
