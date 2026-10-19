# ml-services-base:latest is the CPU build; scripts/build_images.py passes the
# CUDA build of the same base for the GPU variant.
ARG BASE_IMAGE=ml-services-base:latest
FROM ${BASE_IMAGE}

# Default port; overridden at runtime by docker-compose (APP_PORT from .env)
ENV APP_PORT=10005
ARG HOST="0.0.0.0"
ENV HOST=${HOST}

COPY /src/deepmarkpy/plugins/attacks/speech_enhancement_1/requirements.txt .
COPY /src/deepmarkpy/plugins/attacks/speech_enhancement_1/constraints.txt .
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

COPY src/deepmarkpy/plugins/attacks/speech_enhancement_1/app.py .
COPY src/deepmarkpy/plugins/attacks/speech_enhancement_1/inference.py .
COPY src/deepmarkpy/plugins/attacks/speech_enhancement_1/config.json .

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD ["sh", "-c", "exec uvicorn app:app --host=$HOST --port=$APP_PORT"]