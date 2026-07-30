FROM ml-services-base:latest

# Default port; overridden at runtime by docker-compose (APP_PORT from .env)
ENV APP_PORT=9004
ARG HOST="0.0.0.0"
ENV HOST=${HOST}

# Install git for cloning
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Clone and install AWARE package
WORKDIR /app
# Two constraint files: the editable install resolves aware's own exact
# pins (stage 1); requirements.txt later upgrades some of them. Each pip
# step is pinned to the set it resolves.
COPY /src/deepmarkpy/plugins/models/aware/constraints.stage1.txt /app/constraints.stage1.txt
COPY /src/deepmarkpy/plugins/models/aware/constraints.txt /app/constraints.txt
RUN git clone https://github.com/deepmarkpy/aware.git aware && \
    cd aware && git checkout fea9c49e3dfc57a421705ae411adedd80bcc6d09 && \
    pip install --no-cache-dir -e . -c /app/constraints.stage1.txt

# Go back to app directory
WORKDIR /app

# Install service-specific dependencies
COPY /src/deepmarkpy/plugins/models/aware/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -c /app/constraints.txt

# Copy application files
COPY src/deepmarkpy/plugins/models/aware/app.py .
COPY src/deepmarkpy/plugins/models/aware/inference.py .
COPY src/deepmarkpy/plugins/models/aware/config.json .

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT