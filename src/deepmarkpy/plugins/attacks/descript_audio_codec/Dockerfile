FROM ml-services-base:latest

# Default port; overridden at runtime by docker-compose (APP_PORT from .env)
ENV APP_PORT=10008
ARG HOST="0.0.0.0"
ENV HOST=${HOST}

COPY /src/deepmarkpy/plugins/attacks/descript_audio_codec/requirements.txt .
COPY /src/deepmarkpy/plugins/attacks/descript_audio_codec/constraints.txt .
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

COPY src/deepmarkpy/plugins/attacks/descript_audio_codec/app.py .
COPY src/deepmarkpy/plugins/attacks/descript_audio_codec/inference.py .
COPY src/deepmarkpy/plugins/attacks/descript_audio_codec/config.json .

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD ["sh", "-c", "exec uvicorn app:app --host=$HOST --port=$APP_PORT"]
