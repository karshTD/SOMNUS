FROM python:3.11-slim

# No gcc / libpq-dev: psycopg2-BINARY ships prebuilt wheels and needs neither.
# Removing them takes the image from ~1.2GB to ~250MB.
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    SOMNUS_DATA_DIR=/app/data

RUN useradd -m somnus && mkdir -p /app/data /app/skills && chown -R somnus /app
USER somnus

EXPOSE 8501
CMD ["python", "main.py"]
