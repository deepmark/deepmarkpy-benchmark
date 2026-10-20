FROM ml-services-base:latest

# Default port; overridden at runtime by docker-compose (APP_PORT from .env)
ENV APP_PORT=10006
ARG HOST="0.0.0.0"
ENV HOST=${HOST}

COPY /src/deepmarkpy/plugins/attacks/speech_enhancement_2/requirements.txt .
COPY /src/deepmarkpy/plugins/attacks/speech_enhancement_2/constraints.txt .

# clearvoice asks only for torch>=2.0.1, so an unconstrained resolve took the
# default build -- which now ships the CUDA runtime even on arm64. That is
# ~3.5 GB of libraries no container here can reach: compose requests no GPU,
# and torch.cuda.is_available() is False. Installing the +cpu wheel of the
# same version first keeps the numerics and leaves the CUDA stack out.
RUN pip install --no-cache-dir torch==2.13.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu
RUN pip install clearvoice --no-cache-dir -c constraints.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

COPY src/deepmarkpy/plugins/attacks/speech_enhancement_2/app.py .
COPY src/deepmarkpy/plugins/attacks/speech_enhancement_2/inference.py .
COPY src/deepmarkpy/plugins/attacks/speech_enhancement_2/config.json .

# Expose the port defined by the build argument/environment variable
EXPOSE ${APP_PORT}

# Run the FastAPI server using Uvicorn
CMD ["sh", "-c", "exec uvicorn app:app --host=$HOST --port=$APP_PORT"]