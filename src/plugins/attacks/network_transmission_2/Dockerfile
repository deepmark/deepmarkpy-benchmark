FROM python:3.10-slim

ENV APP_PORT=10021
ENV HOST="0.0.0.0"

# iproute2: tc netem
# libopus-dev + libopus0: opuslib needs the C library
# libsndfile1: for soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    iproute2 \
    libopus-dev \
    libopus0 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/plugins/attacks/network_transmission_2/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/plugins/attacks/network_transmission_2 .

EXPOSE 10021

CMD uvicorn app:app --host=$HOST --port=$APP_PORT
