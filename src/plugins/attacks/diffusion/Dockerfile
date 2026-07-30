FROM ml-services-base:latest

# Default port; overridden at runtime by docker-compose (APP_PORT from .env)
ENV APP_PORT=10002
ARG HOST="0.0.0.0"
ENV HOST=${HOST}

COPY /src/plugins/attacks/diffusion/requirements.txt .
COPY /src/plugins/attacks/diffusion/constraints.txt .
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Explicit COPYs: the host-side client (attack.py) is not part of the image
# (REORG_PLAN §5.1 — stop copying host-side files while touched).
COPY src/plugins/attacks/diffusion/app.py .
COPY src/plugins/attacks/diffusion/inference.py .
COPY src/plugins/attacks/diffusion/config.json .

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT