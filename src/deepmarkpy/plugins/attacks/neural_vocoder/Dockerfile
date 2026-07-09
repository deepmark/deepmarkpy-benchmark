FROM ml-services-base:latest AS builder

# Default port; overridden at runtime by docker-compose (APP_PORT from .env)
ENV APP_PORT=10004
ARG HOST="0.0.0.0"
ENV HOST=${HOST}

# Handle BigVGAN in builder stage
RUN git clone https://github.com/NVIDIA/BigVGAN.git 
COPY /src/deepmarkpy/plugins/attacks/neural_vocoder/big_vgan_requirements.txt BigVGAN/requirements.txt
RUN pip install --no-cache-dir -r BigVGAN/requirements.txt

FROM ml-services-base:latest

# Copy BigVGAN from builder
COPY --from=builder /app/BigVGAN /app/BigVGAN

COPY src/deepmarkpy/utils /app/BigVGAN/app_utils
COPY src/deepmarkpy/plugins/attacks/neural_vocoder/config.json /app/BigVGAN/config.json

COPY /src/deepmarkpy/plugins/attacks/neural_vocoder/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/deepmarkpy/plugins/attacks/neural_vocoder .

WORKDIR /app/BigVGAN

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT