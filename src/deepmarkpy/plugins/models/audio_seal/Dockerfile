FROM ml-services-base:latest

# Default port; overridden at runtime by docker-compose (APP_PORT from .env)
ENV APP_PORT=5001
ARG HOST="0.0.0.0"
ENV HOST=${HOST}

# Only install service-specific dependencies not in base
COPY /src/deepmarkpy/plugins/models/audio_seal/requirements.txt .
COPY /src/deepmarkpy/plugins/models/audio_seal/constraints.txt .
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

COPY src/deepmarkpy/plugins/models/audio_seal/app.py .
COPY src/deepmarkpy/plugins/models/audio_seal/inference.py .
COPY src/deepmarkpy/plugins/models/audio_seal/config.json .

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD ["sh", "-c", "exec uvicorn app:app --host=$HOST --port=$APP_PORT"]