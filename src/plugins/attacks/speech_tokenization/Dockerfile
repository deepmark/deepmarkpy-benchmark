FROM ml-services-base:latest

# Default port; overridden at runtime by docker-compose (APP_PORT from .env)
ENV APP_PORT=10003
ARG HOST="0.0.0.0"
ENV HOST=${HOST}

COPY /src/plugins/attacks/speech_tokenization/requirements.txt .
COPY /src/plugins/attacks/speech_tokenization/xcodec_requirements.txt .
COPY /src/plugins/attacks/speech_tokenization/constraints.txt .

RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt && \
    pip install --no-cache-dir xcodec2==0.1.3 --no-deps && \
    pip install --no-cache-dir -r xcodec_requirements.txt -c constraints.txt

# Explicit COPYs: the host-side client (attack.py) is not part of the image
# (REORG_PLAN §5.1 — stop copying host-side files while touched).
COPY src/plugins/attacks/speech_tokenization/app.py .
COPY src/plugins/attacks/speech_tokenization/inference.py .
COPY src/plugins/attacks/speech_tokenization/config.json .

RUN python -c "from xcodec2.modeling_xcodec2 import XCodec2Model; XCodec2Model.from_pretrained('HKUST-Audio/xcodec2')"

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT