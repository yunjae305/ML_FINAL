FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data/beverages.csv data/sample_input.csv data/sample_beverage_log.csv ./data/
COPY mlruns/cardiocare_model.joblib ./mlruns/cardiocare_model.joblib

CMD ["python", "src/inference.py", "--input", "data/sample_input.csv", "--beverages", "data/sample_beverage_log.csv", "--output", "mlruns/docker_prediction.json"]
