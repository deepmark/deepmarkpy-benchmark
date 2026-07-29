FROM ml-services-base:latest

WORKDIR /app

# Default port; overridden at runtime by docker-compose (APP_PORT from .env)
ENV APP_PORT=8001
ARG HOST="0.0.0.0"
ENV HOST=${HOST}

COPY /src/deepmarkpy/plugins/models/wavmark/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/deepmarkpy/plugins/models/wavmark .

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT