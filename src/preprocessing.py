from __future__ import annotations

import hashlib
import json
import urllib.request
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
REPORT_FIGURE_PATHS = {
    "target_distribution": MLRUNS_DIR / "report_target_distribution.png",
    "continuous_boxplot": MLRUNS_DIR / "report_continuous_boxplot.png",
    "model_comparison": MLRUNS_DIR / "report_model_comparison.png",
}
REPORT_MEDICAL_DISCLAIMER = (
    "주의: 이 리포트와 예측 결과는 전문적인 의학 진단이 아니며, 치료 결정은 보건실 또는 의료진과 상의해야 합니다."
)

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
PARALLEL_JOBS = 1


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def ensure_uci_data(force: bool = False) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_DATA_PATH.exists() and not force and sha256_file(RAW_DATA_PATH) == UCI_SHA256:
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


def prepare_model_data(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    empty_columns = [column for column in frame.columns if frame[column].isna().all()]
    if empty_columns:
        frame = frame.drop(columns=empty_columns)
    frame = frame.drop_duplicates().reset_index(drop=True)
    return frame[FEATURE_COLUMNS + [TARGET_COLUMN]]


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
                        n_jobs=PARALLEL_JOBS,
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


def _report_page(pdf: PdfPages, title: str, lines: list[str], font_prop=None, image_path: Path | None = None) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.94, title, fontsize=18, weight="bold", va="top", fontproperties=font_prop)
    y = 0.89
    text_bottom = 0.36 if image_path and image_path.exists() else 0.08
    for line in lines:
        if y < text_bottom:
            break
        fig.text(0.08, y, line, fontsize=9.4, va="top", wrap=True, fontproperties=font_prop)
        y -= 0.029
    if image_path and image_path.exists():
        image = plt.imread(str(image_path))
        ax = fig.add_axes([0.10, 0.06, 0.80, 0.28])
        ax.imshow(image)
        ax.axis("off")
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


def _eda_summary_lines() -> list[str]:
    df = prepare_model_data(load_heart_data())
    target_distribution = df[TARGET_COLUMN].value_counts(normalize=True).sort_index()
    missing_by_column = df.isna().sum()
    missing_nonzero = {column: int(value) for column, value in missing_by_column.items() if int(value) > 0}
    outlier_rows = []
    for column in CONTINUOUS_FEATURES:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((df[column] < lower) | (df[column] > upper)).sum())
        outlier_rows.append(f"{column} {count}")
    return [
        f"데이터셋: UCI processed Cleveland, {len(df)} rows, {len(FEATURE_COLUMNS)} clinical features.",
        f"타깃 분포: normal {target_distribution.get(0, 0):.3f}, heart disease {target_distribution.get(1, 0):.3f}.",
        "평가 지표 선택: 클래스 균형과 false negative 비용을 고려해 balanced accuracy와 recall을 핵심 지표로 둡니다.",
        f"결측값: 전체 {int(missing_by_column.sum())}, 열별 {missing_nonzero if missing_nonzero else '없음'}.",
        f"중복 행: {int(df.duplicated().sum())}; 빈 컬럼은 학습 전 제거합니다.",
        f"IQR 이상치 수: {', '.join(outlier_rows)}.",
        "결측값은 삭제 대신 median 또는 most-frequent imputation으로 처리해 303개 표본을 보존합니다.",
        "이상치는 제거하지 않고 boxplot 근거로 표시만 하며, 임상적으로 의미 있는 고혈압·고콜레스테롤 신호로 남깁니다.",
    ]


def _model_comparison_lines(training: dict) -> list[str]:
    results = training.get("base_model_results", [])
    if not results:
        return ["MLflow 모델 비교: 아직 학습 결과가 없습니다."]
    lines = ["MLflow 모델 비교 표: model | balanced accuracy | precision | recall | F1 | confusion matrix"]
    for result in results:
        metrics = result.get("metrics", {})
        lines.append(
            f"{result.get('name', 'model')}: "
            f"{metrics.get('balanced_accuracy', 0):.4f} | "
            f"{metrics.get('precision', 0):.4f} | "
            f"{metrics.get('recall', 0):.4f} | "
            f"{metrics.get('f1', 0):.4f} | "
            f"{result.get('confusion_matrix', [])}"
        )
    return lines


def _drift_lines(monitoring: dict) -> list[str]:
    drift_rows = monitoring.get("drift_report", [])
    lines = [
        f"원본 테스트셋 Balanced Accuracy: {monitoring.get('baseline_balanced_accuracy', '미실행')}",
        f"드리프트 테스트셋 Balanced Accuracy: {monitoring.get('shifted_balanced_accuracy', '미실행')}",
        f"성능 변화량: {monitoring.get('balanced_accuracy_delta', '미실행')}",
        "monitor.py는 chol 평균을 이동시키고 oldpeak 변동을 키운 합성 테스트셋을 만듭니다.",
        "각 연속형 특성에는 scipy.stats.ks_2samp를 적용하며 p < 0.05인 특성을 drift로 플래그합니다.",
    ]
    lines.extend(
        f"{row['feature']}: KS={row.get('ks_statistic', 0):.4f}, p-value={row['p_value']:.4g}, drift={row['drift_flag']}"
        for row in drift_rows
    )
    return lines


def _build_report_sections(training: dict, monitoring: dict) -> list[tuple[str, list[str]]]:
    final_model = training.get("final_model", {})
    metrics = final_model.get("metrics", {})
    threshold = final_model.get("threshold", "학습 전")
    selected = final_model.get("selected_original_features", [])
    confusion = final_model.get("confusion_matrix", [])

    return [
        (
            "1. 문제 정의와 사용 목적",
            [
                "목적: UCI Heart Disease 데이터 기반 심장 위험 신호 예측을 개인 사용자가 이해할 수 있는 대응 가이드로 바꿉니다.",
                "핵심 원칙: CardioCare는 inform, not decide 원칙의 상담 준비 보조 도구이며 전문적인 의학 진단이나 치료 결정을 대신하지 않습니다.",
                REPORT_MEDICAL_DISCLAIMER,
                "데이터: data/processed.cleveland.data의 UCI Cleveland 데이터를 사용합니다.",
                "타깃 이진화: UCI target 0은 정상, 1~4는 심장병 있음으로 변환합니다.",
                "출력: 심장 위험 확률, action level, 대응 행동, 상담 질문, 의료진에게 보여줄 visit summary입니다.",
                "응급 증상 입력이 true이면 모델 확률보다 사람의 안전을 우선하여 urgent로 라우팅합니다.",
            ],
        ),
        (
            "2. EDA 핵심 결과",
            _eda_summary_lines()
            + [
                "notebooks/01_eda_preprocessing.ipynb는 head(), info(), describe(), target 분포, 결측값, 중복, boxplot, IQR 이상치 탐지를 포함합니다.",
                "그림 1 캡션: target 분포가 완전 불균형은 아니지만 false negative 비용 때문에 recall과 balanced accuracy를 함께 봅니다.",
                "그림 2 캡션: 연속형 boxplot은 trestbps, chol, oldpeak의 극단값을 보여 주며 제거보다 검증과 보존을 선택한 근거입니다.",
            ],
        ),
        (
            "3. 전처리 결정",
            [
                "연속형 변수는 median imputation 후 StandardScaler를 적용합니다.",
                "범주형 코드 변수는 most-frequent imputation 후 OneHotEncoder를 적용합니다.",
                "chol [0, 600] 같은 임상 범위 검증을 추론 전 단계에 포함합니다.",
                "이상치는 무조건 제거하지 않고, 의료적으로 의미 있는 극단값일 수 있어 분석 대상으로 남깁니다.",
                "빈 컬럼은 학습 전 제거하고 중복 행은 prepare_model_data에서 제거합니다.",
                "스케일러, 임퓨터, OneHotEncoder, SelectFromModel은 train_test_split 이후 sklearn Pipeline 내부에서 fit되어 데이터 누수를 막습니다.",
                "EDA 연결: 결측이 ca와 thal에 소수로 집중되어 삭제보다 대치를 선택했고, target 분포 때문에 단순 accuracy 대신 balanced accuracy와 recall을 사용합니다.",
                "EDA 연결: IQR 이상치가 있는 chol, trestbps, oldpeak는 심혈관 위험 신호일 수 있어 학습에서 제거하지 않고 입력 범위 검증으로 통제합니다.",
            ],
        ),
        (
            "4. MLflow 모델 비교와 최종 선택",
            _model_comparison_lines(training)
            + _format_metrics(metrics)
            + [
                f"Threshold: {threshold}",
                f"Confusion Matrix: {confusion}",
                f"선택된 주요 원본 feature: {selected}",
                "Logistic Regression, SVC, Random Forest를 비교하고, 5-fold CV와 GridSearchCV를 수행했습니다.",
                "최종 모델은 false negative를 줄이기 위해 recall과 balanced accuracy를 함께 보는 threshold tuned Logistic Regression입니다.",
                "SVC는 balanced accuracy가 더 높지만 Logistic Regression threshold tuned는 같은 recall과 같은 false negative 수를 유지하면서 threshold 해석이 쉽습니다.",
                "임상적으로 해석 가능한 특성: age, trestbps, chol, thalach, oldpeak, cp, exang은 혈압·지질·운동 반응·흉통 신호와 직접 연결됩니다.",
                "MLflow에는 파라미터, 지표, confusion_matrix.json, selected_features.json, 모델 artifact, model_family tag를 기록합니다.",
                "그림 3 캡션: 모델 비교 그래프는 balanced accuracy와 recall을 나란히 보여 주며, 최종 선택이 recall 중심 안전 전략임을 확인합니다.",
            ],
        ),
        (
            "5. 테스트와 패키징",
            [
                "unittest는 예측 shape, 확률 범위와 행별 합, 임상 범위 검증, 고정 시드 결정론을 검사합니다.",
                "추가 테스트는 음료 로그 요약, urgent/consult action routing, 상담 질문, visit summary 필드를 검증합니다.",
                "Dockerfile은 requirements.txt 기반 의존성 설치 후 저장된 모델과 코드를 복사하고 샘플 CSV batch inference를 실행합니다.",
                "GitHub Actions는 push와 pull_request에서 의존성 설치, python src/train.py, python src/monitor.py, python -m unittest, Docker build/run을 실행합니다.",
                "Feature store 후보: cholesterol. 추론 입력, 임상 해석, drift monitoring 모두에서 핵심 특성이기 때문입니다.",
                "Model registry metadata 후보: tuned threshold. false negative와 보수적 상담 권고의 tradeoff를 결정하기 때문입니다.",
            ],
        ),
        (
            "6. 드리프트 결과와 재학습 계획",
            _drift_lines(monitoring)
            + [
                "재학습 정책: p < 0.05 drift가 반복되고 balanced accuracy 하락이 확인되면 새 라벨 검토 후 재학습합니다.",
                "정기 재학습은 월 1회 후보로 두되, drift trigger가 있으면 우선순위를 높입니다.",
                "로깅 근거: 일반 inference는 actual label이 없을 수 있지만 monitor 경로는 test label을 actual label로 함께 저장해 예측값과 정답을 대조합니다.",
                "폭주 피드백 루프 위험: 모델 권고가 사용자 행동과 라벨 수집을 바꾸면 편향된 새 데이터가 누적될 수 있습니다.",
                "Human-in-the-loop: consult, urgent, low-confidence, drift-triggered retraining 후보는 보건실 또는 의료진 검토를 거칩니다.",
            ],
        ),
        (
            "7. 서빙 선택",
            [
                "서빙 선택: Model-as-a-Service를 선택합니다.",
                "지연 시간: 현재 입력은 13개 임상 특성의 batch 또는 단건 추론이므로 서버 호출 지연은 허용 가능한 범위입니다.",
                "개인정보(PHI): 온디바이스보다 중앙 서버가 민감정보 집중 위험이 있으므로 최소 입력, 로그 비식별화, 접근 통제가 필요합니다.",
                "업데이트 주기: 모델 threshold와 drift 대응 정책을 빠르게 교체해야 하므로 서버형 배포가 온디바이스보다 관리하기 쉽습니다.",
                "운영 방식: Docker 이미지는 저장된 모델 artifact로 추론하며, inference.py는 timestamp, model version, input shape, prediction을 파일 로그로 남깁니다.",
            ],
        ),
        (
            "8. 한계, 윤리, AI 도구 사용 공개",
            [
                "한계: Cleveland 데이터는 규모가 작고 오래된 임상 데이터라 실제 대학생 집단이나 지역 의료 환경을 대표하지 못할 수 있습니다.",
                "윤리: 예측 결과는 전문적인 의학 진단이 아닌 상담 준비 자료이며, emergency warning symptoms는 모델 확률보다 우선합니다.",
                "한 주가 더 주어진다면 calibration curve, 외부 검증 데이터, clinician review checklist, 더 명확한 PHI 로그 정책을 추가합니다.",
                "AI 도구 사용 공개: ChatGPT/Codex를 보일러플레이트 정리, 디버깅, 요구사항 대조에 사용했으며 최종 코드와 결과 해석의 책임은 제출자가 집니다.",
                "근거 자료: CDC Heart Disease Prevention, CDC Risk Factors, AHA Life's Essential 8, AHA Heart Attack Warning Signs.",
            ],
        ),
    ]


def _write_report_figures(training: dict) -> dict[str, Path]:
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    df = prepare_model_data(load_heart_data())

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    ax.bar(["normal", "heart disease"], counts.values, color=["#52796f", "#c44536"])
    ax.set_title("Target class distribution")
    ax.set_ylabel("Rows")
    for index, value in enumerate(counts.values):
        ax.text(index, value, str(int(value)), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(REPORT_FIGURE_PATHS["target_distribution"], dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    df[CONTINUOUS_FEATURES].boxplot(ax=ax)
    ax.set_title("Continuous feature boxplots")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(REPORT_FIGURE_PATHS["continuous_boxplot"], dpi=160)
    plt.close(fig)

    results = training.get("base_model_results", [])
    if results:
        names = [result.get("name", "model") for result in results]
        balanced = [result.get("metrics", {}).get("balanced_accuracy", 0.0) for result in results]
        recall = [result.get("metrics", {}).get("recall", 0.0) for result in results]
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        ax.bar(x - 0.18, balanced, width=0.36, label="Balanced accuracy", color="#3f6c8f")
        ax.bar(x + 0.18, recall, width=0.36, label="Recall", color="#d1843f")
        ax.set_title("Model comparison")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.legend()
        fig.tight_layout()
        fig.savefig(REPORT_FIGURE_PATHS["model_comparison"], dpi=160)
        plt.close(fig)

    return {
        "2. EDA 핵심 결과": REPORT_FIGURE_PATHS["target_distribution"],
        "3. 전처리 결정": REPORT_FIGURE_PATHS["continuous_boxplot"],
        "4. MLflow 모델 비교와 최종 선택": REPORT_FIGURE_PATHS["model_comparison"],
    }


def _write_report_pdf(sections: list[tuple[str, list[str]]], training: dict) -> None:
    font_prop = _korean_font_properties()
    plt.rcParams["axes.unicode_minus"] = False
    if font_prop is not None:
        plt.rcParams["font.family"] = font_prop.get_name()
    figure_paths = _write_report_figures(training)
    with PdfPages(REPORT_PATH) as pdf:
        for title, lines in sections:
            _report_page(pdf, title, lines, font_prop, figure_paths.get(title))


def _html_escape(value) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _report_metric_svg(training: dict, monitoring: dict) -> str:
    final_model = training.get("final_model", {})
    metrics = final_model.get("metrics", {})
    rows = [
        ("Balanced Accuracy", metrics.get("balanced_accuracy", 0.0)),
        ("Recall", metrics.get("recall", 0.0)),
        ("F1", metrics.get("f1", 0.0)),
    ]
    row_svg = []
    y = 80
    for name, value in rows:
        numeric = max(0.0, min(1.0, _as_float(value)))
        width = round(numeric * 260, 1)
        row_svg.append(
            f'<text x="32" y="{y}" font-size="13" fill="#596579">{_html_escape(name)}</text>'
            f'<rect x="170" y="{y - 13}" width="260" height="14" rx="7" fill="#d8dee6"/>'
            f'<rect x="170" y="{y - 13}" width="{width}" height="14" rx="7" fill="#3f6c8f"/>'
            f'<text x="446" y="{y}" font-size="13" font-weight="700" fill="#1f2933">{numeric:.4f}</text>'
        )
        y += 34
    delta = _as_float(monitoring.get("balanced_accuracy_delta", 0.0))
    delta_width = round(min(1.0, abs(delta)) * 260, 1)
    delta_color = "#c44536" if delta < 0 else "#52796f"
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="560" height="230" viewBox="0 0 560 230" role="img" aria-label="CardioCare model metric graph">'
        '<rect width="560" height="230" rx="14" fill="#f8fafc"/>'
        '<text x="32" y="38" font-size="20" font-weight="700" fill="#1f2933">Model performance and drift</text>'
        + "".join(row_svg)
        + '<text x="32" y="196" font-size="13" fill="#596579">Drift delta</text>'
        '<rect x="170" y="183" width="260" height="14" rx="7" fill="#d8dee6"/>'
        f'<rect x="170" y="183" width="{delta_width}" height="14" rx="7" fill="{delta_color}"/>'
        f'<text x="446" y="196" font-size="13" font-weight="700" fill="#1f2933">{delta:.4f}</text>'
        '</svg>'
    )


def _write_report_html(sections: list[tuple[str, list[str]]], training: dict, monitoring: dict) -> None:
    final_model = training.get("final_model", {})
    metrics = final_model.get("metrics", {})
    drift_rows = monitoring.get("drift_report", [])
    metric_svg = _report_metric_svg(training, monitoring)

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
    .svg-panel svg {{
      display: block;
      width: 100%;
      max-width: 560px;
      height: auto;
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
      {REPORT_MEDICAL_DISCLAIMER}
    </div>
    <section class="svg-panel">
      <h2>SVG Summary Graph</h2>
      {metric_svg}
    </section>
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
    _write_report_pdf(sections, training)
    _write_report_html(sections, training, monitoring)
