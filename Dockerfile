FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# --- THE FIX IS HERE ---
# We force install an older version of setuptools before everything else
RUN pip install --no-cache-dir "setuptools<82.0.0" "wheel"
# -----------------------

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
