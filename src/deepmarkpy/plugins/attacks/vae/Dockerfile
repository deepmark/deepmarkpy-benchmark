# ml-services-base:latest is the CPU build; scripts/build_images.py passes the
# CUDA build of the same base for the GPU variant.
ARG BASE_IMAGE=ml-services-base:latest
FROM ${BASE_IMAGE}

# Default port; overridden at runtime by docker-compose (APP_PORT from .env)
ENV APP_PORT=10001
ARG HOST="0.0.0.0"
ENV HOST=${HOST}

COPY /src/deepmarkpy/plugins/attacks/vae/requirements.txt .
COPY /src/deepmarkpy/plugins/attacks/vae/constraints.txt .
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

COPY src/deepmarkpy/plugins/attacks/vae/app.py .
COPY src/deepmarkpy/plugins/attacks/vae/inference.py .
COPY src/deepmarkpy/plugins/attacks/vae/config.json .

# Expose the service port
EXPOSE 10001

# Run the FastAPI server using Uvicorn
CMD ["sh", "-c", "exec uvicorn app:app --host=$HOST --port=$APP_PORT"]