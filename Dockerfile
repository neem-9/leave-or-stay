# ---- Stage 1: install dependencies and get the model ----
FROM python:3.12.15-slim-trixie AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
ENV PYTHONPATH=/install/lib/python3.12/site-packages
COPY src/ src/
COPY data/ data/
COPY artifacts/ artifacts/
ARG GIT_SHA=local
# Lab 3 onwards: the promotion gate exports the @champion model to artifacts/.
# Lab 2: artifacts/ is empty, so the image trains the model itself.
RUN if [ -f artifacts/model.joblib ]; then \
      mkdir -p models && cp artifacts/model.joblib artifacts/metadata.json models/ && \
      python -c "import json,sys; p='models/metadata.json'; m=json.load(open(p)); m['model_version']=sys.argv[1][:7]; json.dump(m,open(p,'w'))" "$GIT_SHA"; \
    else \
      GIT_SHA=$GIT_SHA python -m src.train.train; \
    fi

# ---- Stage 2: a small runtime image, no build tools, not root ----
FROM python:3.12.15-slim-trixie
LABEL org.opencontainers.image.title="ai381-churn-starter" \
      org.opencontainers.image.description="AI381 Machine Learning Operations churn API starter"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_DIR=/app/models
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder /install /usr/local
COPY --from=builder /build/models models/
COPY src/ src/
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"]
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
