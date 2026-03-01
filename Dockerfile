FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/app/src:/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY Code /app/Code
COPY src /app/src
COPY ui /app/ui
COPY main.py /app/main.py
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md

CMD ["python", "main.py", "--help"]
