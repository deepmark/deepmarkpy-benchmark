FROM ml-services-base:latest

# Default port; overridden at runtime by docker-compose (APP_PORT from .env)
ENV APP_PORT=7001
ARG HOST="0.0.0.0"
ENV HOST=${HOST}

COPY /src/plugins/models/silent_cipher/requirements.txt .
COPY /src/plugins/models/silent_cipher/constraints.txt .
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

COPY src/plugins/models/silent_cipher/app.py .
COPY src/plugins/models/silent_cipher/inference.py .
COPY src/plugins/models/silent_cipher/config.json .

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT