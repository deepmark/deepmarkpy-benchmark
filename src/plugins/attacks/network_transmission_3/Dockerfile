FROM python:3.10-slim

ENV APP_PORT=10022
ENV HOST="0.0.0.0"

# - iproute2: tc netem
# - libopus-dev / libopus0: opuslib needs the Opus C library
# - libsndfile1: for soundfile
# - swig, build-essential, python3-dev, libc++-dev: required to build
#   python-webrtc-audio-processing from source
# - git: to clone the build sources
RUN apt-get update && apt-get install -y --no-install-recommends \
    iproute2 \
    libopus-dev \
    libopus0 \
    libsndfile1 \
    swig \
    build-essential \
    python3-dev \
    libc++-dev \
    libc++abi-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/plugins/attacks/network_transmission_3/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build python-webrtc-audio-processing from the same fork v1 uses, with
# the pinned upstream commit known to compile cleanly.
RUN git clone https://github.com/nesovicelena/python-webrtc-audio-processing.git /tmp/python-webrtc-audio-processing && \
    cd /tmp/python-webrtc-audio-processing && \
    git clone https://gitlab.freedesktop.org/pulseaudio/webrtc-audio-processing.git webrtc-audio-processing && \
    cd webrtc-audio-processing && git checkout ee8cfef49b8417c2f0ba65a249d2ee8c360d19ab && cd .. && \
    python setup.py build && \
    python setup.py install && \
    rm -rf /tmp/python-webrtc-audio-processing

COPY src/plugins/attacks/network_transmission_3 .

EXPOSE 10022

CMD uvicorn app:app --host=$HOST --port=$APP_PORT
