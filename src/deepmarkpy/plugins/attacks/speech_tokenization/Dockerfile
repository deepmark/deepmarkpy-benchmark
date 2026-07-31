FROM ml-services-base:latest

# Default port; overridden at runtime by docker-compose (APP_PORT from .env)
ENV APP_PORT=10003
ARG HOST="0.0.0.0"
ENV HOST=${HOST}

COPY /src/deepmarkpy/plugins/attacks/speech_tokenization/requirements.txt .
COPY /src/deepmarkpy/plugins/attacks/speech_tokenization/xcodec_requirements.txt .
COPY /src/deepmarkpy/plugins/attacks/speech_tokenization/constraints.txt .

RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt && \
    pip install --no-cache-dir xcodec2==0.1.3 --no-deps && \
    pip install --no-cache-dir -r xcodec_requirements.txt -c constraints.txt

COPY src/deepmarkpy/plugins/attacks/speech_tokenization/app.py .
COPY src/deepmarkpy/plugins/attacks/speech_tokenization/inference.py .
COPY src/deepmarkpy/plugins/attacks/speech_tokenization/config.json .

# Weights baked at build, pinned so a rebuild cannot pick up a different
# upstream checkpoint.
RUN python -c "from xcodec2.modeling_xcodec2 import XCodec2Model; XCodec2Model.from_pretrained('HKUST-Audio/xcodec2', revision='e412427ed30f0cf9d5e3c95562113deb10a32d03')"

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD uvicorn app:app --host=$HOST --port=$APP_PORT