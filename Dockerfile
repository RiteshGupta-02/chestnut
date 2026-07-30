# Start from an official Python base image
# slim = smaller size, no unnecessary packages
FROM python:3.10-slim

# Set working directory inside the container
# All subsequent commands run from here
WORKDIR /app

# Copy requirements first — before copying code
# WHY: Docker caches each layer. If you copy code first,
# every code change triggers a full pip install.
# Copying requirements first means pip install only reruns
# when requirements.txt changes — saves minutes on every rebuild.
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code
COPY src/ ./src/
COPY api/ ./api/
COPY src/checkpoint/checkpoint_epoch8.tar ./src/checkpoint/

# Expose port 8080 — GCP Cloud Run expects this port specifically
EXPOSE 8080

# Start the FastAPI server
# host 0.0.0.0 = accept connections from outside the container
# port 8080 = what Cloud Run expects
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]