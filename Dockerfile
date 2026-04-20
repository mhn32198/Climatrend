# Switching to 3.11-slim is the safest move for pandasai in 2026
FROM python:3.11-slim

WORKDIR /app

# Install system-level compilers for climate data processing
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# CRITICAL: Fix for the February 2026 pkg_resources removal.
# We install the last stable version of setuptools and force-reinstall 
# the legacy pkg_resources bridge.
RUN pip install --upgrade pip
RUN pip install --no-cache-dir "setuptools<82.0.0" "wheel"

# Now install your specific climate and AI libraries
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

# Ensure Streamlit is configured for the Google Cloud environment
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
