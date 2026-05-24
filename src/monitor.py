from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference import load_model_bundle, log_inference, predict  # noqa: E402
from preprocessing import (  # noqa: E402
    CONTINUOUS_FEATURES,
    FEATURE_COLUMNS,
    MLRUNS_DIR,
    MONITORING_SUMMARY_PATH,
    TARGET_COLUMN,
    TEST_SET_PATH,
    TRAIN_REFERENCE_PATH,
    write_json,
    write_report_pdf,
)


def score(bundle: dict, frame: pd.DataFrame, threshold: float) -> float:
    probabilities = bundle["pipeline"].predict_proba(frame[FEATURE_COLUMNS])[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return float(balanced_accuracy_score(frame[TARGET_COLUMN], predictions))


def run_monitoring() -> dict:
    if not TRAIN_REFERENCE_PATH.exists() or not TEST_SET_PATH.exists():
        raise FileNotFoundError("Run `python src/train.py` before `python src/monitor.py`.")

    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_model_bundle()
    threshold = float(bundle.get("threshold", 0.5))
    train_reference = pd.read_csv(TRAIN_REFERENCE_PATH)
    test_set = pd.read_csv(TEST_SET_PATH)

    shifted = test_set.copy()
    shifted["chol"] = shifted["chol"] * 1.10 + 30
    shifted["oldpeak"] = shifted["oldpeak"] * 1.15

    drift_rows = []
    for feature in CONTINUOUS_FEATURES:
        statistic, p_value = ks_2samp(train_reference[feature].dropna(), shifted[feature].dropna())
        drift_rows.append(
            {
                "feature": feature,
                "ks_statistic": float(statistic),
                "p_value": float(p_value),
                "drift_flag": bool(p_value < 0.05),
            }
        )
    pd.DataFrame(drift_rows).to_csv(MLRUNS_DIR / "drift_report.csv", index=False)

    baseline_balanced_accuracy = score(bundle, test_set, threshold)
    shifted_balanced_accuracy = score(bundle, shifted, threshold)

    sample_result = predict(test_set[FEATURE_COLUMNS].head(5), model_bundle=bundle)
    log_inference(
        sample_result,
        input_shape=(5, len(FEATURE_COLUMNS)),
        actual=test_set[TARGET_COLUMN].head(5).tolist(),
        log_path=MLRUNS_DIR / "inference_monitor.log",
    )

    base_time = datetime.now(timezone.utc) - timedelta(days=4)
    rows = []
    for index, shift in enumerate([0, 10, 20, 30, 40]):
        version = test_set.copy()
        version["chol"] = version["chol"] * (1 + shift / 300) + shift
        rows.append(
            {
                "timestamp": (base_time + timedelta(days=index)).isoformat(),
                "chol_shift": shift,
                "balanced_accuracy": score(bundle, version, threshold),
            }
        )
    timeseries = pd.DataFrame(rows)
    timeseries.to_csv(MLRUNS_DIR / "monitoring_timeseries.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(timeseries["timestamp"], timeseries["balanced_accuracy"], marker="o")
    ax.set_title("Balanced accuracy under synthetic cholesterol drift")
    ax.set_xlabel("Synthetic timestamp")
    ax.set_ylabel("Balanced accuracy")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(MLRUNS_DIR / "monitoring_timeseries.png", dpi=160)
    plt.close(fig)

    summary = {
        "baseline_balanced_accuracy": baseline_balanced_accuracy,
        "shifted_balanced_accuracy": shifted_balanced_accuracy,
        "balanced_accuracy_delta": shifted_balanced_accuracy - baseline_balanced_accuracy,
        "drift_report": drift_rows,
    }
    write_json(MONITORING_SUMMARY_PATH, summary)
    write_report_pdf()
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_monitoring()
