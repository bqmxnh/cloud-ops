FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY retrain_incremental.py retrain_incremental.py

CMD ["python", "retrain_incremental.py"]
