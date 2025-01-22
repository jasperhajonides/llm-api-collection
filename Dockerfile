# Use a lightweight official Python image
FROM python:3.9-slim

# Prevent Python from writing pyc files and set stdout to be unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create and use /app as the working directory
WORKDIR /app

# Copy the requirements file first
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code into /app
COPY . .

# By default, run the main.py script
CMD ["python", "main.py"]
