FROM ml-services-base:latest

# Default port; overridden at runtime by docker-compose (APP_PORT from .env)
ENV APP_PORT=10007
ARG HOST="0.0.0.0"
ENV HOST=${HOST}

COPY /src/plugins/attacks/encodec/requirements.txt .
COPY /src/plugins/attacks/encodec/constraints.txt .
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

COPY src/plugins/attacks/encodec/app.py .
COPY src/plugins/attacks/encodec/inference.py .
COPY src/plugins/attacks/encodec/config.json .

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT
