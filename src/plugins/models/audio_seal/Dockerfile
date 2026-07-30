FROM ml-services-base:latest

# Default port; overridden at runtime by docker-compose (APP_PORT from .env)
ENV APP_PORT=5001
ARG HOST="0.0.0.0"
ENV HOST=${HOST}

# Only install service-specific dependencies not in base
COPY /src/plugins/models/audio_seal/requirements.txt .
COPY /src/plugins/models/audio_seal/constraints.txt .
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Explicit COPYs: the host-side client (model.py) is not part of the image
# (REORG_PLAN §5.1 — stop copying host-side files while touched).
COPY src/plugins/models/audio_seal/app.py .
COPY src/plugins/models/audio_seal/inference.py .
COPY src/plugins/models/audio_seal/config.json .

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT