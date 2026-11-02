FROM mambaorg/micromamba:2.9.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    MPLBACKEND=Agg

USER root
# ONNX Runtime uses OpenMP; pydub/Gradio use ffmpeg for media handling.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app/docs /app/flashrank_model \
    && chown -R $MAMBA_USER:$MAMBA_USER /app

WORKDIR /app
USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER appl-kgraph.yml /tmp/appl-kgraph.yml
RUN micromamba install -y -n base -f /tmp/appl-kgraph.yml \
    && micromamba clean --all --yes

COPY --chown=$MAMBA_USER:$MAMBA_USER graph/ ./graph/

EXPOSE 7860
CMD ["python", "graph/app.py"]
