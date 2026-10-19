# ml-services-base:latest is the CPU build; scripts/build_images.py passes the
# CUDA build of the same base for the GPU variant.
ARG BASE_IMAGE=ml-services-base:latest
FROM ${BASE_IMAGE}

# Default port; overridden at runtime by docker-compose (APP_PORT from .env)
ENV APP_PORT=7010
ARG HOST="0.0.0.0"
ENV HOST=${HOST}

# Only install service-specific dependencies not in base
COPY /src/deepmarkpy/plugins/models/perth/requirements.txt .
COPY /src/deepmarkpy/plugins/models/perth/constraints.txt .
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

COPY src/deepmarkpy/plugins/models/perth/app.py .
COPY src/deepmarkpy/plugins/models/perth/inference.py .
COPY src/deepmarkpy/plugins/models/perth/config.json .

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD ["sh", "-c", "exec uvicorn app:app --host=$HOST --port=$APP_PORT"]