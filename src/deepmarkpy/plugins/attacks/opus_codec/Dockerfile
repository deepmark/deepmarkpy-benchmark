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

# Install the deepmarkpy package (modules only; runtime deps are the
# image's own frozen set)
COPY pyproject.toml README.md LICENSE /tmp/deepmarkpy-pkg/
COPY src/deepmarkpy /tmp/deepmarkpy-pkg/src/deepmarkpy
RUN pip install --no-cache-dir --no-deps /tmp/deepmarkpy-pkg && rm -rf /tmp/deepmarkpy-pkg

COPY src/deepmarkpy/plugins/attacks/opus_codec/requirements.txt .
COPY src/deepmarkpy/plugins/attacks/opus_codec/constraints.txt .
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

COPY src/deepmarkpy/plugins/attacks/opus_codec/app.py .
COPY src/deepmarkpy/plugins/attacks/opus_codec/inference.py .
COPY src/deepmarkpy/plugins/attacks/opus_codec/config.json .

EXPOSE 10023

CMD ["sh", "-c", "exec uvicorn app:app --host=$HOST --port=$APP_PORT"]
