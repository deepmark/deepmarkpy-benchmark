FROM ml-services-base:latest

WORKDIR /app

# Default port; overridden at runtime by docker-compose (APP_PORT from .env)
ENV APP_PORT=8001
ARG HOST="0.0.0.0"
ENV HOST=${HOST}

COPY /src/plugins/models/wavmark/requirements.txt .
COPY /src/plugins/models/wavmark/constraints.txt .
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Explicit COPYs: the host-side client (model.py) is not part of the image
# (REORG_PLAN §5.1 — stop copying host-side files while touched).
COPY src/plugins/models/wavmark/app.py .
COPY src/plugins/models/wavmark/inference.py .
COPY src/plugins/models/wavmark/config.json .

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT