# Imagen de un workflow exportado. Construir desde la raíz del repo:
#   docker build -t mi-workflow --build-arg WORKFLOW=workflows/<nombre> .
#   docker run -p 8000:8000 --env-file workflows/<nombre>/.env mi-workflow
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY runtime ./runtime
ARG WORKFLOW
COPY ${WORKFLOW} ./workflow
ENV WORKFLOW_DIR=/app/workflow PORT=8000

EXPOSE 8000
CMD ["sh", "-c", "python -m runtime serve \"$WORKFLOW_DIR\" --port \"$PORT\""]
