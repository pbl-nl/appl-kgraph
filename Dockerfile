FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    MPLBACKEND=Agg

WORKDIR /app

# ONNX Runtime uses OpenMP; pydub/Gradio use ffmpeg for media handling.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install -r requirements.txt

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/docs /app/flashrank_model \
    && chown -R app:app /app

COPY --chown=app:app graph/ ./graph/

USER app
EXPOSE 7860

CMD ["python", "graph/app.py"]
