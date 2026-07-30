FROM ml-services-base:latest

# Default port; overridden at runtime by docker-compose (APP_PORT from .env)
ENV APP_PORT=7010
ARG HOST="0.0.0.0"
ENV HOST=${HOST}

# Only install service-specific dependencies not in base
COPY /src/plugins/models/perth/requirements.txt .
COPY /src/plugins/models/perth/constraints.txt .
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

COPY src/plugins/models/perth/app.py .
COPY src/plugins/models/perth/inference.py .
COPY src/plugins/models/perth/config.json .

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT