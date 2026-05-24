from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MLRUNS_DIR = ROOT_DIR / "mlruns"
RAW_DATA_PATH = DATA_DIR / "processed.cleveland.data"
MODEL_PATH = MLRUNS_DIR / "cardiocare_model.joblib"
TRAIN_REFERENCE_PATH = DATA_DIR / "train_reference.csv"
TEST_SET_PATH = DATA_DIR / "test_set.csv"
TRAINING_SUMMARY_PATH = MLRUNS_DIR / "training_summary.json"
MONITORING_SUMMARY_PATH = MLRUNS_DIR / "monitoring_summary.json"
REPORT_PATH = ROOT_DIR / "report.pdf"
REPORT_HTML_PATH = ROOT_DIR / "report.html"

UCI_URL = (
    "http://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
UCI_SHA256 = "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8"

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
RAW_TARGET_COLUMN = "target_raw"
TARGET_COLUMN = "target"
CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]
CLINICAL_RANGES = {
    "age": (0, 120),
    "trestbps": (0, 300),
    "chol": (0, 600),
    "thalach": (0, 250),
    "oldpeak": (-5, 10),
    "ca": (0, 3),
}
RANDOM_STATE = 42
TEST_SIZE = 0.2


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def ensure_uci_data(force: bool = False) -> Path:
    """Create data/processed.cleveland.data from the submitted local dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_DATA_PATH.exists() and not force and sha256_file(RAW_DATA_PATH) == UCI_SHA256:
        return RAW_DATA_PATH

    local_sources = [
        ROOT_DIR / "heart+disease.zip",
        DATA_DIR / "heart+disease.zip",
    ]
    for zip_path in local_sources:
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as archive:
                data = archive.read("processed.cleveland.data")
            if sha256_bytes(data) != UCI_SHA256:
                raise ValueError("Checksum mismatch for processed.cleveland.data inside heart+disease.zip")
            RAW_DATA_PATH.write_bytes(data)
            return RAW_DATA_PATH

    extracted_sources = [
        ROOT_DIR / "heart+disease" / "processed.cleveland.data",
        DATA_DIR / "heart+disease" / "processed.cleveland.data",
    ]
    for extracted_path in extracted_sources:
        if extracted_path.exists():
            data = extracted_path.read_bytes()
            if sha256_bytes(data) != UCI_SHA256:
                raise ValueError(f"Checksum mismatch for {extracted_path}")
            RAW_DATA_PATH.write_bytes(data)
            return RAW_DATA_PATH

    with urllib.request.urlopen(UCI_URL, timeout=30) as response:
        data = response.read()
    if sha256_bytes(data) != UCI_SHA256:
        raise ValueError("Checksum mismatch for downloaded UCI data")
    RAW_DATA_PATH.write_bytes(data)
    return RAW_DATA_PATH


def load_heart_data() -> pd.DataFrame:
    data_path = ensure_uci_data()
    columns = FEATURE_COLUMNS + [RAW_TARGET_COLUMN]
    df = pd.read_csv(data_path, header=None, names=columns, na_values="?")
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[TARGET_COLUMN] = (df[RAW_TARGET_COLUMN] > 0).astype(int)
    return df[FEATURE_COLUMNS + [TARGET_COLUMN]]


def load_beverages(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_DIR / "beverages.csv"
    df = pd.read_csv(path)
    for column in ["size_ml", "caffeine_mg", "calories_kcal", "sugar_g"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    df["drink_id"] = df["drink_id"].astype(str)
    return df


def search_beverages(query: str, beverages: pd.DataFrame | None = None, limit: int = 10) -> pd.DataFrame:
    beverages = load_beverages() if beverages is None else beverages.copy()
    query = (query or "").strip().lower()
    if not query:
        return beverages.head(limit)
    haystack = (
        beverages["name"].str.lower()
        + " "
        + beverages["brand"].str.lower()
        + " "
        + beverages["category"].str.lower()
    )
    return beverages[haystack.str.contains(query, regex=False)].head(limit)


def summarize_beverage_log(log: pd.DataFrame | list[dict] | None) -> dict:
    beverages = load_beverages()
    if log is None:
        log_df = pd.DataFrame(columns=["drink_id", "quantity"])
    elif isinstance(log, pd.DataFrame):
        log_df = log.copy()
    else:
        log_df = pd.DataFrame(log)

    if log_df.empty:
        return {"daily_caffeine_mg": 0.0, "daily_calories_kcal": 0.0, "daily_sugar_g": 0.0}

    if "quantity" not in log_df.columns:
        log_df["quantity"] = 1
    log_df["drink_id"] = log_df["drink_id"].astype(str)
    log_df["quantity"] = pd.to_numeric(log_df["quantity"], errors="coerce").fillna(1)
    merged = log_df.merge(beverages, on="drink_id", how="left", validate="many_to_one")
    if merged["name"].isna().any():
        missing = merged.loc[merged["name"].isna(), "drink_id"].tolist()
        raise ValueError(f"Unknown drink_id values: {missing}")

    return {
        "daily_caffeine_mg": round(float((merged["caffeine_mg"] * merged["quantity"]).sum()), 2),
        "daily_calories_kcal": round(float((merged["calories_kcal"] * merged["quantity"]).sum()), 2),
        "daily_sugar_g": round(float((merged["sugar_g"] * merged["quantity"]).sum()), 2),
    }


def caffeine_alert(caffeine_mg: float, sugar_g: float) -> str:
    if caffeine_mg >= 400:
        return "high"
    if caffeine_mg >= 300 or sugar_g >= 50:
        return "watch"
    if caffeine_mg >= 200 or sugar_g >= 25:
        return "moderate"
    return "low"


def as_feature_frame(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, dict):
        frame = pd.DataFrame([data])
    else:
        frame = pd.DataFrame(data, columns=FEATURE_COLUMNS)

    for column in FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame[FEATURE_COLUMNS]
    for column in FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def validate_clinical_ranges(data, allow_missing: bool = True) -> pd.DataFrame:
    frame = as_feature_frame(data)
    errors = []
    for column, (lower, upper) in CLINICAL_RANGES.items():
        values = frame[column]
        checked = values.dropna() if allow_missing else values
        bad_mask = (checked < lower) | (checked > upper)
        if bad_mask.any():
            errors.append(f"{column} outside [{lower}, {upper}]: {checked[bad_mask].tolist()}")
        if not allow_missing and values.isna().any():
            errors.append(f"{column} contains missing values")
    if errors:
        raise ValueError("; ".join(errors))
    return frame


class ClinicalRangeValidator(BaseEstimator, TransformerMixin):
    def __init__(self, allow_missing: bool = True):
        self.allow_missing = allow_missing

    def fit(self, X, y=None):
        validate_clinical_ranges(X, allow_missing=self.allow_missing)
        return self

    def transform(self, X):
        return validate_clinical_ranges(X, allow_missing=self.allow_missing)


def build_preprocessor() -> ColumnTransformer:
    continuous_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("continuous", continuous_pipeline, CONTINUOUS_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_model_pipeline(estimator, use_selector: bool = True) -> Pipeline:
    steps = [
        ("range_validator", ClinicalRangeValidator(allow_missing=True)),
        ("preprocess", build_preprocessor()),
    ]
    if use_selector:
        steps.append(
            (
                "selector",
                SelectFromModel(
                    RandomForestClassifier(
                        n_estimators=200,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                    threshold="median",
                ),
            )
        )
    steps.append(("model", estimator))
    return Pipeline(steps)


def get_transformed_feature_names(fitted_pipeline: Pipeline) -> np.ndarray:
    names = fitted_pipeline.named_steps["preprocess"].get_feature_names_out()
    if "selector" in fitted_pipeline.named_steps:
        names = names[fitted_pipeline.named_steps["selector"].get_support()]
    return names


def selected_original_features(feature_names) -> list[str]:
    originals = []
    for name in feature_names:
        raw = str(name).split("__", 1)[-1]
        original = raw.split("_", 1)[0]
        if original not in originals:
            originals.append(original)
    return originals


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _korean_font_properties():
    for font_path in [
        Path("C:/Windows/Fonts/NotoSansKR-VF.ttf"),
        Path("C:/Windows/Fonts/malgun.ttf"),
    ]:
        if font_path.exists():
            return font_manager.FontProperties(fname=str(font_path))
    return None


def _report_page(pdf: PdfPages, title: str, lines: list[str], font_prop=None) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.94, title, fontsize=18, weight="bold", va="top", fontproperties=font_prop)
    y = 0.89
    for line in lines:
        fig.text(0.08, y, line, fontsize=10.5, va="top", wrap=True, fontproperties=font_prop)
        y -= 0.035
    pdf.savefig(fig)
    plt.close(fig)


def _format_metrics(metrics: dict) -> list[str]:
    if not metrics:
        return ["모델 지표: 아직 학습 결과가 없습니다."]
    return [
        f"Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.4f}",
        f"Precision: {metrics.get('precision', 0):.4f}",
        f"Recall: {metrics.get('recall', 0):.4f}",
        f"F1: {metrics.get('f1', 0):.4f}",
    ]


def _build_report_sections(training: dict, monitoring: dict) -> list[tuple[str, list[str]]]:
    final_model = training.get("final_model", {})
    metrics = final_model.get("metrics", {})
    drift_rows = monitoring.get("drift_report", [])
    threshold = final_model.get("threshold", "학습 전")
    selected = final_model.get("selected_original_features", [])
    confusion = final_model.get("confusion_matrix", [])

    return [
        (
            "CardioCare Action Navigator",
            [
                "목적: UCI Heart Disease 데이터 기반 심장 위험 신호 예측을 개인 사용자가 이해할 수 있는 대응 가이드로 바꿉니다.",
                "핵심 원칙: 이 서비스는 진단하거나 치료를 결정하지 않습니다. inform, not decide 원칙의 상담 준비 보조 도구입니다.",
                "데이터: 제출된 heart+disease.zip 또는 data/heart+disease/processed.cleveland.data의 processed.cleveland.data를 사용합니다.",
                "타깃 이진화: UCI target 0은 정상, 1~4는 심장병 있음으로 변환합니다.",
                "출력: 심장 위험 확률, action level, 대응 행동, 상담 질문, 의료진에게 보여줄 visit summary를 제공합니다.",
            ],
        ),
        (
            "EDA 및 전처리",
            [
                "노트북은 head(), info(), describe(), target 분포, 결측값, 중복, IQR 기반 이상치 탐지를 포함합니다.",
                "타깃 분포를 확인해 단순 accuracy뿐 아니라 balanced accuracy와 recall을 주요 지표로 사용합니다.",
                "연속형 변수는 median imputation 후 StandardScaler를 적용합니다.",
                "범주형 코드 변수는 most-frequent imputation 후 OneHotEncoder를 적용합니다.",
                "chol [0, 600] 같은 임상 범위 검증을 추론 전 단계에 포함합니다.",
                "이상치는 무조건 제거하지 않고, 의료적으로 의미 있는 극단값일 수 있어 분석 대상으로 남깁니다.",
            ],
        ),
        (
            "모델 학습 및 MLflow",
            _format_metrics(metrics)
            + [
                f"Threshold: {threshold}",
                f"Confusion Matrix: {confusion}",
                f"선택된 주요 원본 feature: {selected}",
                "Logistic Regression, SVC, Random Forest를 비교하고, 5-fold CV와 GridSearchCV를 수행합니다.",
                "최종 모델은 심장 위험 신호를 놓치는 false negative를 줄이기 위해 recall을 중요하게 봅니다.",
                "MLflow에는 파라미터, 지표, confusion matrix, 선택 feature, 모델 artifact, model_family tag를 기록합니다.",
            ],
        ),
        (
            "Action Navigator 결과",
            [
                "heart_probability_percent: 사용자가 이해하기 쉬운 백분율 예측값입니다.",
                "action_level: low, watch, consult, urgent 중 하나를 반환합니다.",
                "response_actions: 사용자가 지금 확인할 사항과 다음 행동을 제시합니다.",
                "consultation_questions: 보건실 또는 의료진 상담 때 물어볼 질문을 생성합니다.",
                "visit_summary: 확률, 등급, 주요 요인, 핵심 수치, disclaimer를 담은 상담용 요약 카드입니다.",
                "응급 증상 입력이 true이면 모델 확률과 무관하게 urgent로 라우팅합니다.",
            ],
        ),
        (
            "모니터링 및 드리프트",
            [
                f"원본 테스트셋 Balanced Accuracy: {monitoring.get('baseline_balanced_accuracy', '미실행')}",
                f"드리프트 테스트셋 Balanced Accuracy: {monitoring.get('shifted_balanced_accuracy', '미실행')}",
                f"성능 변화량: {monitoring.get('balanced_accuracy_delta', '미실행')}",
                "monitor.py는 chol 분포를 인위적으로 이동시키고 scipy.stats.ks_2samp로 연속형 feature drift를 감지합니다.",
            ]
            + [
                f"{row['feature']}: p-value={row['p_value']:.4g}, drift={row['drift_flag']}"
                for row in drift_rows
            ],
        ),
        (
            "배포, CI, 윤리",
            [
                "Dockerfile은 python src/train.py 이후 생성된 모델 artifact로 batch inference를 실행할 수 있게 구성되어 있습니다.",
                "GitHub Actions는 push마다 python -m unittest를 실행합니다.",
                "Human-in-the-loop: consult, urgent, low-confidence 결과는 보건실 또는 의료진 검토를 권장합니다.",
                "Feature store 후보: cholesterol. 추론과 drift monitoring 모두에서 사용되기 때문입니다.",
                "Model registry metadata 후보: tuned threshold. false negative tradeoff를 결정하는 핵심 값이기 때문입니다.",
                "근거 자료: CDC Heart Disease Prevention, CDC Risk Factors, AHA Life's Essential 8, AHA Heart Attack Warning Signs.",
            ],
        ),
    ]


def _write_report_pdf(sections: list[tuple[str, list[str]]]) -> None:
    font_prop = _korean_font_properties()
    plt.rcParams["axes.unicode_minus"] = False
    if font_prop is not None:
        plt.rcParams["font.family"] = font_prop.get_name()
    with PdfPages(REPORT_PATH) as pdf:
        for title, lines in sections:
            _report_page(pdf, title, lines, font_prop)


def _html_escape(value) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _write_report_html(sections: list[tuple[str, list[str]]], training: dict, monitoring: dict) -> None:
    final_model = training.get("final_model", {})
    metrics = final_model.get("metrics", {})
    drift_rows = monitoring.get("drift_report", [])

    metric_cards = "".join(
        f"<div class='metric'><span>{_html_escape(name)}</span><strong>{value:.4f}</strong></div>"
        for name, value in [
            ("Balanced Accuracy", metrics.get("balanced_accuracy", 0.0)),
            ("Recall", metrics.get("recall", 0.0)),
            ("F1", metrics.get("f1", 0.0)),
            ("Threshold", float(final_model.get("threshold", 0.0) or 0.0)),
        ]
    )
    drift_table = "".join(
        "<tr>"
        f"<td>{_html_escape(row['feature'])}</td>"
        f"<td>{row['ks_statistic']:.4f}</td>"
        f"<td>{row['p_value']:.4g}</td>"
        f"<td>{_html_escape(row['drift_flag'])}</td>"
        "</tr>"
        for row in drift_rows
    )
    sections_html = "".join(
        "<section>"
        f"<h2>{_html_escape(title)}</h2>"
        + "".join(f"<p>{_html_escape(line)}</p>" for line in lines)
        + "</section>"
        for title, lines in sections
    )
    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CardioCare Action Navigator Report</title>
  <style>
    body {{
      margin: 0;
      background: #f6f7f9;
      color: #1f2933;
      font-family: "Noto Sans KR", "Malgun Gothic", "Apple SD Gothic Neo", sans-serif;
      line-height: 1.62;
    }}
    main {{
      max-width: 960px;
      margin: 0 auto;
      padding: 40px 24px 64px;
    }}
    header {{
      background: #173b57;
      color: white;
      padding: 36px 40px;
      border-radius: 8px;
    }}
    h1, h2 {{
      letter-spacing: 0;
      margin: 0 0 16px;
    }}
    section {{
      background: white;
      border: 1px solid #d8dee6;
      border-radius: 8px;
      padding: 24px 28px;
      margin-top: 20px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin-top: 20px;
    }}
    .metric {{
      background: white;
      border: 1px solid #d8dee6;
      border-radius: 8px;
      padding: 16px;
    }}
    .metric span {{
      display: block;
      color: #596579;
      font-size: 13px;
    }}
    .metric strong {{
      display: block;
      font-size: 24px;
      margin-top: 4px;
    }}
    .notice {{
      background: #fff7e6;
      border: 1px solid #f5c26b;
      border-radius: 8px;
      padding: 16px 18px;
      margin-top: 20px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      background: white;
    }}
    th, td {{
      border: 1px solid #d8dee6;
      padding: 10px;
      text-align: left;
    }}
    th {{
      background: #edf2f7;
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>CardioCare Action Navigator</h1>
      <p>심장 위험 신호 예측을 개인 대응 가이드와 상담 준비 리포트로 변환하는 End-to-End ML 시스템</p>
    </header>
    <div class="metrics">{metric_cards}</div>
    <div class="notice">
      이 리포트는 의료 진단서가 아닙니다. 예측 결과는 보건실 또는 의료진 상담을 준비하기 위한 참고 자료입니다.
    </div>
    {sections_html}
    <section>
      <h2>KS Drift Report</h2>
      <table>
        <thead><tr><th>Feature</th><th>KS statistic</th><th>p-value</th><th>Drift flag</th></tr></thead>
        <tbody>{drift_table}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""
    REPORT_HTML_PATH.write_text(html, encoding="utf-8")


def write_report_pdf() -> None:
    training = read_json(TRAINING_SUMMARY_PATH)
    monitoring = read_json(MONITORING_SUMMARY_PATH)
    sections = _build_report_sections(training, monitoring)
    _write_report_pdf(sections)
    _write_report_html(sections, training, monitoring)
