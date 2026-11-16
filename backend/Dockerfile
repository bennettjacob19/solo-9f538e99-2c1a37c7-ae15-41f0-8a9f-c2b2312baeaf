FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /srv

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /var/lib/embedding-backend \
    && chown -R appuser:appuser /var/lib/embedding-backend
USER appuser

# Injected mode/config is persisted here so a compose restart cannot silently
# turn an "accepted then dropped" backend back into a healthy one.
ENV MOCK_STATE_FILE=/var/lib/embedding-backend/config.json
VOLUME ["/var/lib/embedding-backend"]

EXPOSE 9000
# One worker: the mock's isolation state (busy slot, counters, injected mode)
# lives in memory and is only meaningful within a single process.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000", "--workers", "1"]
