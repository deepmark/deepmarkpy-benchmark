FROM python:3.10-slim

ENV APP_PORT=10023
ENV HOST="0.0.0.0"

# - opus-tools: opusenc / opusdec CLI
# - libopus0: Opus runtime library
# - libsndfile1: for soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    opus-tools \
    libopus0 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/plugins/attacks/opus_codec/requirements.txt .
COPY src/plugins/attacks/opus_codec/constraints.txt .
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

COPY src/plugins/attacks/opus_codec .

EXPOSE 10023

CMD uvicorn app:app --host=$HOST --port=$APP_PORT
