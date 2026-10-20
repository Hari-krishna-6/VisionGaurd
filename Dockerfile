FROM python:3.12.8-slim-bookworm

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --gid 10001 visionguard \
    && useradd --uid 10001 --gid visionguard --create-home visionguard \
    && mkdir -p /app/artifacts/heatmaps \
    && chown -R visionguard:visionguard /app
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1 torchvision==0.20.1
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1 \
    VISIONGUARD_DATASET_ROOT=/data/visa_prepared/1cls \
    VISIONGUARD_CATEGORY=candle \
    VISIONGUARD_MODEL_PATH=/app/artifacts/model.npz \
    VISIONGUARD_HEATMAP_DIR=/app/artifacts/heatmaps \
    VISIONGUARD_DEVICE=cpu
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
