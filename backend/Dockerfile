FROM python:3.12-slim

WORKDIR /app
COPY backend /app
RUN pip install --no-cache-dir .

ENV PYTHONPATH=/app
ENV EVAL_DATA_DIR=/data
EXPOSE 8000

CMD ["uvicorn", "eval_factory.main:app", "--host", "0.0.0.0", "--port", "8000"]
