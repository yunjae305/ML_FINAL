# CardioCare 기계학습 기말과제

CardioCare는 UCI Heart Disease 데이터를 사용해 심장병 위험 신호를 예측하고, 그 결과를 상담 준비용 행동 가이드로 변환하는 CLI/Docker 기반 ML 시스템입니다. 이 프로젝트는 전문적인 의학 진단 도구가 아니라 `inform, not decide` 원칙을 따르는 의사결정 보조 예시입니다.

## 프로젝트 구성

```text
data/
  processed.cleveland.data
  beverages.csv
  sample_input.csv
  sample_beverage_log.csv
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
  cardiocare_model.joblib
  training_summary.json
  monitoring_summary.json
  drift_report.csv
  monitoring_timeseries.csv
  monitoring_timeseries.png
Dockerfile
requirements.txt
.github/workflows/ci.yml
report.pdf
README.md
```

## 전체 재현 절차

채점자는 저장소를 clone한 뒤 아래 순서로 학습, 모니터링, 테스트, Docker 추론을 재현할 수 있습니다.

```bash
pip install -r requirements.txt
python src/train.py
python src/monitor.py
python -m unittest
docker build -t cardiocare:1.0 .
docker run --rm cardiocare:1.0
```

샘플 입력에 대해 추론만 실행할 때는 다음 명령을 사용합니다.

```bash
python src/inference.py --input data/sample_input.csv --beverages data/sample_beverage_log.csv
```

## 데이터와 전처리

제출 데이터는 `data/processed.cleveland.data`입니다. 코드는 실행 시 SHA-256 체크섬을 확인합니다.

```text
a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8
```

타깃 값은 다음처럼 이진화합니다.

```text
0 -> 정상
1,2,3,4 -> 심장병 있음
```

전처리는 결측값을 삭제하지 않고 수치형과 범주형 특성에 맞는 파이프라인으로 처리합니다. `data/beverages.csv`, `data/sample_input.csv`, `data/sample_beverage_log.csv`는 Docker와 예시 추론을 위한 작은 샘플 파일입니다.

## 학습과 MLflow

`python src/train.py`는 전처리 파이프라인과 모델을 학습하고, MLflow에 실험 정보를 기록합니다. 최종 모델 번들은 아래 위치에 저장됩니다.

```text
mlruns/cardiocare_model.joblib
```

학습 과정에서는 Logistic Regression, SVC, Random Forest를 비교하고, 각 실험에 balanced accuracy, precision, recall, F1, `confusion_matrix.json`, `selected_features.json`, 모델 artifact, `model_family tag`를 기록합니다.

현재 저장된 최종 모델은 `Logistic Regression threshold tuned`입니다.

```text
threshold: 0.48
balanced accuracy: 0.9064
precision: 0.8438
recall: 0.9643
f1: 0.9000
confusion matrix: [[28, 5], [1, 27]]
```

SVC는 balanced accuracy가 더 높지만 최종 선택은 recall을 유지하면서 설명 가능한 선형 모델과 낮은 false negative 수를 우선한 threshold-tuned Logistic Regression입니다.

MLflow UI는 다음 명령으로 확인할 수 있습니다.

```bash
mlflow ui --backend-store-uri ./mlruns
```

## 추론과 행동 가이드

`src/inference.py`는 저장된 모델로 배치 추론을 실행하고, 예측 확률을 상담 준비용 정보로 변환합니다. 출력에는 위험 단계, confidence level, 주요 위험 특성, 카페인/당 섭취 요약, 상담 질문, visit summary, SVG risk summary가 포함됩니다.

응급 증상 입력이 true이면 모델 확률보다 사용자 안전을 우선하여 `urgent` 행동 단계로 라우팅합니다.

## 모니터링과 드리프트

`python src/monitor.py`는 추론 로그를 생성하고, 테스트 데이터의 연속형 특성을 인위적으로 이동시킨 뒤 `scipy.stats.ks_2samp`로 데이터 드리프트를 확인합니다. 주요 산출물은 다음과 같습니다.

```text
mlruns/monitoring_summary.json
mlruns/drift_report.csv
mlruns/monitoring_timeseries.csv
mlruns/monitoring_timeseries.png
```

현재 모니터링 결과는 baseline balanced accuracy 0.9064에서 shifted balanced accuracy 0.8333으로 감소했으며, `chol`, `thalach`, `oldpeak`에서 drift flag가 발생했습니다.

## 테스트와 패키징

`python -m unittest`는 전처리, 예측 shape, 확률 범위, 임상 범위 검증, 드리프트 헬퍼, README/CI/Docker 제출 조건을 확인합니다. GitHub Actions는 push와 pull request마다 학습, 모니터링, unittest, Docker build/run을 실행합니다.

Docker 이미지는 실행에 필요한 `src/`, 샘플 데이터, 최종 모델만 복사하도록 구성되어 있습니다.

## 최종 보고서

최종 보고서는 `report.pdf` 단일 파일입니다. 학습과 모니터링 명령을 실행하면 `report.pdf`가 다시 생성됩니다. 로컬 확인용 `report.html`도 생성될 수 있지만 제출 대상은 아닙니다.

## 안전 및 윤리

이 시스템의 출력은 전문적인 의학 진단이 아니며 치료 결정의 단독 근거로 사용하면 안 됩니다. consult, urgent, low-confidence, drift-triggered retraining 후보는 사람 검토를 거치는 Human-in-the-loop 흐름으로 다룹니다.

참고 자료:

- https://www.cdc.gov/heart-disease/prevention/index.html
- https://www.cdc.gov/heart-disease/risk-factors/index.html
- https://www.heart.org/en/healthy-living/healthy-lifestyle/lifes-essential-8
- https://www.stroke.org/en/health-topics/heart-attack/warning-signs-of-a-heart-attack
