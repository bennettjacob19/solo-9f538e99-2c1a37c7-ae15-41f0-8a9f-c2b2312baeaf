FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /srv

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /var/lib/embedding-gateway \
    && chown -R appuser:appuser /var/lib/embedding-gateway
USER appuser

# The compose file mounts a named volume here so the quarantine record
# survives `docker compose down/up`.  A fresh named volume inherits the
# ownership of this directory from the image, so appuser can write to it.
ENV GATEWAY_STATE_FILE=/var/lib/embedding-gateway/state.json
VOLUME ["/var/lib/embedding-gateway"]

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
