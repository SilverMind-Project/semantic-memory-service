FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for building python packages (if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements/pyproject files and the app package (required by hatchling to build the wheel)
COPY pyproject.toml README.md ./
COPY app ./app

# Install dependencies
RUN pip install --no-cache-dir .

# Copy the rest of the application
COPY . .

# Expose the service port
EXPOSE 8400

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8400/health || exit 1

# Command to run the application
CMD ["python", "-m", "app.run"]
