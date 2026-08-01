FROM python:3.10-slim

ENV APP_PORT=10020
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

# Install the deepmarkpy package (modules only; runtime deps are the
# image's own frozen set). This image does not build on ml-services-base,
# so it needs its own copy the way opus_codec does.
COPY pyproject.toml README.md LICENSE /tmp/deepmarkpy-pkg/
COPY src/deepmarkpy /tmp/deepmarkpy-pkg/src/deepmarkpy
RUN pip install --no-cache-dir --no-deps /tmp/deepmarkpy-pkg && rm -rf /tmp/deepmarkpy-pkg

COPY src/deepmarkpy/plugins/attacks/network_transmission/requirements.txt .
COPY src/deepmarkpy/plugins/attacks/network_transmission/constraints.txt .
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Build the Python binding from the DeepMark fork, which carries the build
# fixes this image needs; both it and the vendored upstream sources are
# pinned so the image is reproducible.
RUN git clone https://github.com/deepmark/python-webrtc-audio-processing.git /tmp/python-webrtc-audio-processing && \
    cd /tmp/python-webrtc-audio-processing && \
    git checkout d8231843956bba2a04a4b95777aa5424ecb2939c && \
    git clone https://gitlab.freedesktop.org/pulseaudio/webrtc-audio-processing.git webrtc-audio-processing && \
    cd webrtc-audio-processing && git checkout ee8cfef49b8417c2f0ba65a249d2ee8c360d19ab && cd .. && \
    python setup.py build && \
    python setup.py install && \
    rm -rf /tmp/python-webrtc-audio-processing

COPY src/deepmarkpy/plugins/attacks/network_transmission .

EXPOSE 10020

CMD ["sh", "-c", "exec uvicorn app:app --host=$HOST --port=$APP_PORT"]
