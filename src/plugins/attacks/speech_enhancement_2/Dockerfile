FROM ml-services-base:latest

# Default port; overridden at runtime by docker-compose (APP_PORT from .env)
ENV APP_PORT=10006
ARG HOST="0.0.0.0"
ENV HOST=${HOST}

COPY /src/plugins/attacks/speech_enhancement_2/requirements.txt .
COPY /src/plugins/attacks/speech_enhancement_2/constraints.txt .
RUN pip install clearvoice --no-cache-dir -c constraints.txt
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Explicit COPYs: the host-side client (attack.py) is not part of the image
# (REORG_PLAN §5.1 — stop copying host-side files while touched).
COPY src/plugins/attacks/speech_enhancement_2/app.py .
COPY src/plugins/attacks/speech_enhancement_2/inference.py .
COPY src/plugins/attacks/speech_enhancement_2/config.json .

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT