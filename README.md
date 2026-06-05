# CardioCare Action Navigator

UCI Heart Disease risk-signal prediction plus a personal action guide and consultation-preparation report.

This project is a CLI/Docker-based ML system, not a web UI application.

The project follows the required structure:

```text
data/
notebooks/
  01_eda_preprocessing.ipynb
src/
  preprocessing.py
  train.py
  inference.py
  monitor.py
tests/
  test_pipeline.py
mlruns/
Dockerfile
requirements.txt
.github/workflows/ci.yml
report.pdf
report.html
README.md
```

## Reproduce From Scratch

```bash
pip install -r requirements.txt
python src/train.py
python src/monitor.py
python -m unittest
docker build -t cardiocare:1.0 .
docker run --rm cardiocare:1.0
```

Optional inference command:

```bash
python src/inference.py --input data/sample_input.csv --beverages data/sample_beverage_log.csv
```

## Reports

Training and monitoring regenerate both report formats:

```text
report.pdf
report.html
```

`report.html` is a standalone Korean HTML report. Open it directly in a browser. The report CSS uses Korean font fallbacks: `Noto Sans KR`, `Malgun Gothic`, `Apple SD Gothic Neo`, and `sans-serif`.

## Dataset

The intended data source is the provided `heart+disease.zip`. The code extracts `processed.cleveland.data` and verifies this checksum:

```text
a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8
```

If the zip has already been extracted, the loader also accepts `heart+disease/processed.cleveland.data` or `data/heart+disease/processed.cleveland.data`. The UCI URL is only a fallback when no local copy exists.

Target conversion:

```text
0 -> normal
1,2,3,4 -> heart disease
```

## What The System Does

The ML model uses the 13 official UCI clinical features. It returns a heart-risk probability and then converts that result into:

- `action_level`: `low`, `watch`, `consult`, or `urgent`
- `response_actions`: non-prescriptive next steps
- `consultation_questions`: questions to ask a campus health center or clinician
- `visit_summary`: a compact report to bring to consultation
- `medical_disclaimer`: explicit "inform, not decide" safety statement

Optional caffeine or energy-drink records are lifestyle context only. They are not used as UCI model inputs.

## Safety Framing

This is not a diagnosis or treatment tool. Emergency warning symptoms override the model score and route the user to urgent help. The report wording follows non-prescriptive public-health guidance from CDC heart-disease prevention/risk-factor resources and AHA heart-attack warning-sign guidance.

Sources:

- https://www.cdc.gov/heart-disease/prevention/index.html
- https://www.cdc.gov/heart-disease/risk-factors/index.html
- https://www.heart.org/en/healthy-living/healthy-lifestyle/lifes-essential-8
- https://www.stroke.org/en/health-topics/heart-attack/warning-signs-of-a-heart-attack

## MLflow

`python src/train.py` creates MLflow runs under `mlruns/`, logs all required metrics and artifacts, and saves the final model bundle to:

```text
mlruns/cardiocare_model.joblib
```

Each model run records params, balanced accuracy, precision, recall, F1, `confusion_matrix.json`, `selected_features.json`, a model artifact, and a `model_family tag`.

To inspect runs:

```bash
mlflow ui --backend-store-uri ./mlruns
```

## Monitoring

`python src/monitor.py` logs inference, creates synthetic cholesterol drift, runs `scipy.stats.ks_2samp`, compares balanced accuracy before/after drift, and writes monitoring artifacts under `mlruns/`.

Key monitoring artifacts:

```text
mlruns/monitoring_summary.json
mlruns/drift_report.csv
mlruns/monitoring_timeseries.csv
mlruns/monitoring_timeseries.png
mlruns/inference_monitor.log
```
