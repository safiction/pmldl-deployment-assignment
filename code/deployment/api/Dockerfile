# Build context is the repository root (needs code/models/features.py too):
#   docker build -f code/deployment/api/Dockerfile -t walmart-api .
FROM python:3.11-slim

WORKDIR /app

COPY code/deployment/api/requirements.txt code/deployment/api/requirements.txt
RUN pip install --no-cache-dir -r code/deployment/api/requirements.txt

COPY code/models/features.py code/models/features.py
COPY code/deployment/api/main.py code/deployment/api/main.py

WORKDIR /app/code/deployment/api
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
