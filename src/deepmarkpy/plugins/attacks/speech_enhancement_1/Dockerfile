FROM ml-services-base:latest

# Default port; overridden at runtime by docker-compose (APP_PORT from .env)
ENV APP_PORT=10005
ARG HOST="0.0.0.0"
ENV HOST=${HOST}

COPY /src/deepmarkpy/plugins/attacks/speech_enhancement_1/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/deepmarkpy/plugins/attacks/speech_enhancement_1 .

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT