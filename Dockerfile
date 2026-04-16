FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for building python packages (if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements/pyproject files
COPY pyproject.toml .

# Install dependencies
# Using pip directly since we're in a Docker container and don't necessarily need poetry's env management
RUN pip install --no-cache-dir .

# Copy the rest of the application
COPY . .

# Expose the service port
EXPOSE 8300

# Command to run the application
CMD ["python", "-m", "app.run"]
