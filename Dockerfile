# Use a slim version of Python to stay under the 0.5GB Free Storage limit
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for climate math
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Cloud Run uses port 8080 by default
EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
