# AI381 Lab 2 – Churn Prediction API

[![CI](https://github.com/hedraababneh/ai381-lab2/actions/workflows/ci.yml/badge.svg)](https://github.com/hedraababneh/ai381-lab2/actions/workflows/ci.yml)

A logistic-regression customer churn model, served with FastAPI, packaged in a multi-stage non-root Docker image, and built, scanned (Trivy) and pushed to GHCR automatically by GitHub Actions.

## Run it (one command)

```bash
docker run -p 8000:8000 ghcr.io/hedraababneh/ai381-lab2@sha256:791826cb45fab0c46955f420d4b8287af4bfb4d4c6922adc676ac64dbaf1874a
```

Then open http://localhost:8000/docs

## Try a prediction

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"tenure_months":2,"monthly_fee":95.5,"contract":"month-to-month","internet_service":"fiber","support_calls":5}'
```

Response: `{"churn":1,"probability":0.9532}`

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check, returns `{"status":"ok"}` |
| POST | `/predict` | Churn prediction (0/1) and probability for one customer |

## Image

- Registry: `ghcr.io/hedraababneh/ai381-lab2`
- Tagged with the full commit SHA on every push to `main`
- Digest: `sha256:791826cb45fab0c46955f420d4b8287af4bfb4d4c6922adc676ac64dbaf1874a`

## CI pipeline

- **Every pull request:** ruff lint → train model → 8 pytest tests → build image → Trivy scan (fails on fixable HIGH/CRITICAL)
- **Push to `main`:** all of the above, then push the image to GHCR tagged with the commit SHA
- All third-party actions are pinned to full commit SHAs
- `main` is protected: changes need a pull request with passing checks

## Develop locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python src/train/train.py
pytest -v
uvicorn src.app.main:app --reload
```