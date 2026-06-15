FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install system dependencies if any, then python requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the serving application and the model directory
COPY prometheus_exporter.py .
COPY model/ ./model/

# Expose the API and metrics port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "prometheus_exporter:app", "--host", "0.0.0.0", "--port", "8000"]
