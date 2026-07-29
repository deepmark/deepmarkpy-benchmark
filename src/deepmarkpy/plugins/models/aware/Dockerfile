FROM ml-services-base:latest

# Default port; overridden at runtime by docker-compose (APP_PORT from .env)
ENV APP_PORT=9004
ARG HOST="0.0.0.0"
ENV HOST=${HOST}

# Install git for cloning
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Clone and install AWARE package
WORKDIR /app
RUN git clone https://github.com/deepmarkpy/aware.git aware && \
    cd aware && pip install --no-cache-dir -e .

# Go back to app directory
WORKDIR /app

# Install service-specific dependencies
COPY /src/deepmarkpy/plugins/models/aware/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/deepmarkpy/plugins/models/aware/app.py .
COPY src/deepmarkpy/plugins/models/aware/model.py .
COPY src/deepmarkpy/plugins/models/aware/config.json .
COPY src/deepmarkpy/utils ./utils

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT