# Self-contained image: installs deps, trains a default checkpoint at build time, serves API.
# Override training length: docker build --build-arg TRAIN_EPOCHS=50000 .
# To skip training, build with a pre-made weights.pt in build context and adjust COPY/RUN below.

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WEIGHTS_PATH=/app/weights.pt

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/names ./data/names
COPY name_classification ./name_classification
COPY api ./api

ARG TRAIN_EPOCHS=12000
RUN python -m name_classification.train \
    --data-dir /app/data/names \
    --epochs ${TRAIN_EPOCHS} \
    -o /app/weights.pt \
    --print-every 4000 \
    --plot-every 1000

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
