FROM ml-services-base:latest

ENV APP_PORT=10030
ARG HOST="0.0.0.0"
ENV HOST=${HOST}

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY src/deepmarkpy/services/nisqa/requirements.txt .
COPY src/deepmarkpy/services/nisqa/constraints.txt .
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

RUN mkdir -p /app/weights && \
    curl -fSL https://raw.githubusercontent.com/gabrielmittag/NISQA/master/weights/nisqa.tar \
    -o /app/weights/nisqa.tar

COPY src/deepmarkpy/services/nisqa/app.py .
COPY src/deepmarkpy/services/nisqa/inference.py .

EXPOSE ${APP_PORT}
CMD ["sh", "-c", "exec uvicorn app:app --host=$HOST --port=$APP_PORT"]
