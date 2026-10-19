# ml-services-base:latest is the CPU build; scripts/build_images.py passes the
# CUDA build of the same base for the GPU variant.
ARG BASE_IMAGE=ml-services-base:latest
FROM ${BASE_IMAGE} AS builder

# Default port; overridden at runtime by docker-compose (APP_PORT from .env)
ENV APP_PORT=10004
ARG HOST="0.0.0.0"
ENV HOST=${HOST}

# Handle BigVGAN in builder stage
RUN git clone https://github.com/NVIDIA/BigVGAN.git && \
    cd BigVGAN && git checkout 7d2b454564a6c7d014227f635b7423881f14bdac
COPY /src/deepmarkpy/plugins/attacks/neural_vocoder/big_vgan_requirements.txt BigVGAN/requirements.txt
COPY /src/deepmarkpy/plugins/attacks/neural_vocoder/constraints.builder.txt .
RUN pip install --no-cache-dir -r BigVGAN/requirements.txt -c constraints.builder.txt

FROM ${BASE_IMAGE}

# ENV does not cross stages: without these, EXPOSE below expands to nothing
# (buildah, and so Podman, refuses to build) and uvicorn gets an empty --host.
ENV APP_PORT=10004
ARG HOST="0.0.0.0"
ENV HOST=${HOST}

# Copy BigVGAN from builder
COPY --from=builder /app/BigVGAN /app/BigVGAN

COPY src/deepmarkpy/plugins/attacks/neural_vocoder/config.json /app/BigVGAN/config.json

COPY /src/deepmarkpy/plugins/attacks/neural_vocoder/requirements.txt .
COPY /src/deepmarkpy/plugins/attacks/neural_vocoder/constraints.txt .
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

COPY src/deepmarkpy/plugins/attacks/neural_vocoder/app.py .
COPY src/deepmarkpy/plugins/attacks/neural_vocoder/inference.py .

WORKDIR /app/BigVGAN

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD ["sh", "-c", "exec uvicorn app:app --host=$HOST --port=$APP_PORT"]