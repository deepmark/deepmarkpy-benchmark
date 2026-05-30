FROM ml-services-base:latest

# Default port; overridden at runtime by docker-compose (APP_PORT from .env)
ENV APP_PORT=10003
ARG HOST="0.0.0.0"
ENV HOST=${HOST}

COPY /src/plugins/attacks/speech_tokenization/requirements.txt .
COPY /src/plugins/attacks/speech_tokenization/xcodec_requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir xcodec2==0.1.3 --no-deps && \
    pip install --no-cache-dir -r xcodec_requirements.txt

COPY src/plugins/attacks/speech_tokenization .

RUN python -c "from xcodec2.modeling_xcodec2 import XCodec2Model; XCodec2Model.from_pretrained('HKUST-Audio/xcodec2')"

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT