FROM ml-services-base:latest

# Default port; overridden at runtime by docker-compose (APP_PORT from .env)
ENV APP_PORT=10001
ARG HOST="0.0.0.0"
ENV HOST=${HOST}

COPY /src/plugins/attacks/vae/requirements.txt .
COPY /src/plugins/attacks/vae/constraints.txt .
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

COPY src/plugins/attacks/vae .

# Expose the service port
EXPOSE 10001

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT