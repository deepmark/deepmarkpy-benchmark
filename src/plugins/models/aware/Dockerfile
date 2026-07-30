FROM ml-services-base:latest

# Default port; overridden at runtime by docker-compose (APP_PORT from .env)
ENV APP_PORT=9004
ARG HOST="0.0.0.0"
ENV HOST=${HOST}

# Install git for cloning
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Clone and install AWARE package
# Commit pinned per REORG_PLAN P0.2 (build reproducibility): upstream main
# HEAD resolved 2026-07-29; no locally-built image existed to recover an
# earlier commit from.
WORKDIR /app
# Two constraint files because this build is path-dependent: the editable
# install resolves aware's own exact pins (stage 1), then requirements.txt
# upgrades some of them (final). Each pip step is pinned to what it resolved
# at capture time (REORG_PLAN P0.2 freeze-in-place).
COPY /src/plugins/models/aware/constraints.stage1.txt /app/constraints.stage1.txt
COPY /src/plugins/models/aware/constraints.txt /app/constraints.txt
RUN git clone https://github.com/deepmarkpy/aware.git aware && \
    cd aware && git checkout fea9c49e3dfc57a421705ae411adedd80bcc6d09 && \
    pip install --no-cache-dir -e . -c /app/constraints.stage1.txt

# Go back to app directory
WORKDIR /app

# Install service-specific dependencies
COPY /src/plugins/models/aware/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -c /app/constraints.txt

# Copy application files
COPY src/plugins/models/aware/app.py .
COPY src/plugins/models/aware/model.py .
COPY src/plugins/models/aware/config.json .
COPY src/utils ./utils

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT