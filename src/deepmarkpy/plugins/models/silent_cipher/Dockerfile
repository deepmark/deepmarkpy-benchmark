FROM ml-services-base:latest

# Default port; overridden at runtime by docker-compose (APP_PORT from .env)
ENV APP_PORT=7001
ARG HOST="0.0.0.0"
ENV HOST=${HOST}

COPY /src/deepmarkpy/plugins/models/silent_cipher/requirements.txt .
COPY /src/deepmarkpy/plugins/models/silent_cipher/constraints.txt .
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# silentcipher pins torch 2.0.0, but nothing re-requests torchaudio, so the
# base image's 2.7.1 survives the downgrade. Its C++ extension is ABI-linked
# to libtorch 2.7, and `import torchaudio` then dies on an undefined symbol.
# Nothing here imports it today; this keeps the pair consistent so nothing
# has to discover that the hard way.
RUN pip install --no-cache-dir torchaudio==2.0.1 -c constraints.txt

COPY src/deepmarkpy/plugins/models/silent_cipher/app.py .
COPY src/deepmarkpy/plugins/models/silent_cipher/inference.py .
COPY src/deepmarkpy/plugins/models/silent_cipher/config.json .

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD ["sh", "-c", "exec uvicorn app:app --host=$HOST --port=$APP_PORT"]