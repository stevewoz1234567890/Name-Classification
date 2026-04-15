# Name classification

Character-level RNN that predicts the likely **language or cultural origin** of a personal name (PyTorch, based on the [official tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)). This repository packages training, a small CLI, a **REST API** that returns the top-N labels with **log-probability scores**, a **Docker** image, and a **Helm** chart for Kubernetes.

## Layout

| Path | Purpose |
|------|---------|
| `data/names/*.txt` | One file per category; each line is a training name |
| `name_classification/` | Model, data loading, training, checkpoint I/O, inference |
| `api/main.py` | FastAPI app (`/classify`, `/health`) |
| `Dockerfile` | Installs dependencies, trains a default checkpoint at **image build** time, runs Uvicorn |
| `helm/name-classification/` | Deployment + Service (+ optional Ingress) |

Training saves a **checkpoint** (weights plus alphabet and category list) so inference does not need the raw text files at runtime.

## Local setup

Use a virtual environment (recommended on Debian/Ubuntu, where system Python is often PEP 668–managed):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
```

## Train

From the repository root:

```bash
python -m name_classification.train --data-dir data/names -o weights.pt
```

Useful flags:

- `--epochs` — default `100000` (tutorial-style); lower for quick experiments
- `--n-hidden`, `--lr`, `--print-every`, `--plot-every`, `--seed`

## CLI inference

After `weights.pt` exists:

```bash
python -m name_classification.predict_cli -w weights.pt -n 5 "Quang"
```

Scores printed are **natural-log probabilities** (the model ends with `LogSoftmax`).

## REST API

Set `WEIGHTS_PATH` if the checkpoint is not at `./weights.pt` or `./weights.pt` next to the repo root:

```bash
export WEIGHTS_PATH="$PWD/weights.pt"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Endpoints

- `GET /health` — liveness; includes `model_loaded`
- `POST /classify` — JSON body, JSON response

**Request**

```json
{
  "name": "Quang",
  "top_n": 5
}
```

**Response**

```json
{
  "name": "Quang",
  "normalized_name": "Quang",
  "predictions": [
    { "label": "Vietnamese", "log_probability": -0.02 },
    { "label": "Chinese", "log_probability": -4.66 }
  ]
}
```

`normalized_name` is the ASCII form used inside the model (diacritics stripped, out-of-alphabet characters removed). `top_n` is capped by the number of categories.

### Example `curl`

```bash
curl -s -X POST http://localhost:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{"name":"Müller","top_n":3}' | jq
```

## Docker

Build (trains inside the image by default; override epoch count if needed):

```bash
docker build -t name-classification:latest .
# Faster build (lower accuracy):
docker build --build-arg TRAIN_EPOCHS=5000 -t name-classification:latest .
```

Run:

```bash
docker run --rm -p 8000:8000 name-classification:latest
```

The container sets `WEIGHTS_PATH=/app/weights.pt`. To use your own checkpoint, extend the image or mount a file and set `WEIGHTS_PATH` accordingly.

## Kubernetes (Helm)

Build and load the image into your cluster (example with [minikube](https://minikube.sigs.k8s.io/docs/)):

```bash
docker build -t name-classification:latest .
minikube image load name-classification:latest
```

Install the chart:

```bash
helm upgrade --install name-classification ./helm/name-classification \
  --set image.repository=name-classification \
  --set image.tag=latest \
  --set image.pullPolicy=Never
```

`pullPolicy: Never` (or `IfNotPresent` with a loaded image) avoids pulling from a registry when using minikube’s Docker daemon or `minikube image load`.

Port-forward to the service:

```bash
kubectl port-forward service/name-classification-name-classification 8000:8000
curl -s http://127.0.0.1:8000/health
```

Enable Ingress in `helm/name-classification/values.yaml` (`ingress.enabled: true`) and configure `ingress.hosts` / TLS for your cluster.

## Notes

- **Scores** are **log-probabilities** (not probabilities). To obtain probabilities, exponentiate and normalize over the full label set (the API returns only the top-N).
- For production, prefer a **pre-trained** checkpoint baked or mounted into the image instead of long training during `docker build`.
- Earlier iterations of this repo kept scripts under a nested `rnn/` folder; everything now lives in `name_classification/` and `data/names/`.
