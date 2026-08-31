FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app

# The image defaults to a non-mutating interface check. Runtime workflows must
# select an explicit producer and configuration.
CMD ["python", "-m", "src.cli", "--help"]
