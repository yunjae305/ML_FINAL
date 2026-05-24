from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing import (  # noqa: E402
    FEATURE_COLUMNS,
    MLRUNS_DIR,
    MODEL_PATH,
    as_feature_frame,
    caffeine_alert,
    summarize_beverage_log,
    validate_clinical_ranges,
)

SYMPTOM_COLUMNS = [
    "symptom_chest_pain_now",
    "symptom_shortness_breath",
    "symptom_fainting",
    "symptom_radiating_pain",
]

MEDICAL_DISCLAIMER = (
    "This result is not a medical diagnosis and should not be used as the only basis "
    "for treatment decisions. It is a risk-signal and consultation-preparation aid."
)


def load_model_bundle(model_path: Path = MODEL_PATH) -> dict:
    if not Path(model_path).exists():
        raise FileNotFoundError("Model not found. Run `python src/train.py` first.")
    return joblib.load(model_path)


def predict_proba(features, model_bundle: dict | None = None):
    bundle = load_model_bundle() if model_bundle is None else model_bundle
    frame = validate_clinical_ranges(as_feature_frame(features), allow_missing=True)
    return bundle["pipeline"].predict_proba(frame)


def risk_tier(probability: float) -> str:
    if probability >= 0.65:
        return "clinical_review_recommended"
    if probability >= 0.35:
        return "check_recommended"
    return "low_signal"


def confidence_level(frame: pd.DataFrame, probability: float) -> str:
    missing_ratio = float(frame.isna().mean(axis=1).iloc[0])
    if missing_ratio >= 0.35:
        return "low"
    if missing_ratio >= 0.15 or 0.45 <= probability <= 0.55:
        return "medium"
    return "high"


def _truthy(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return bool(value)


def extract_symptoms(features) -> dict[str, bool]:
    if isinstance(features, pd.DataFrame):
        row = features.iloc[0].to_dict() if not features.empty else {}
    elif isinstance(features, dict):
        row = features
    else:
        row = {}
    return {column: _truthy(row.get(column, False)) for column in SYMPTOM_COLUMNS}


def determine_action_level(probability: float, symptoms: dict[str, bool], confidence: str) -> str:
    if any(symptoms.values()):
        return "urgent"
    if probability >= 0.65:
        return "consult"
    if probability >= 0.35 or confidence == "low":
        return "watch"
    return "low"


def build_response_actions(action_level: str, top_risk_factors: list[str], caffeine_alert: str) -> list[str]:
    if action_level == "urgent":
        actions = [
            "If warning symptoms are happening now, call 119 or seek emergency medical help immediately.",
            "Do not wait for the model score to change; urgent symptoms override the prediction.",
            "Show the visit summary and recent measurements to emergency staff or a clinician.",
        ]
    elif action_level == "consult":
        actions = [
            "Schedule a campus health-center or clinician consultation and bring the generated summary.",
            "Recheck any unusual blood pressure, cholesterol, heart-rate, or chest-pain-related inputs before the visit.",
            "Ask whether additional evaluation is needed based on your symptoms and measured values.",
        ]
    elif action_level == "watch":
        actions = [
            "Review the entered health-profile values and correct any missing or uncertain measurements.",
            "Monitor for chest pain, shortness of breath, fainting, or pain spreading to the arm, jaw, back, or shoulder.",
            "Consider a non-urgent health-center consultation if the same risk signal repeats.",
        ]
    else:
        actions = [
            "Keep this result as a baseline and update it when new checkup values are available.",
            "Maintain routine prevention habits: blood pressure awareness, cholesterol awareness, sleep, activity, and avoiding tobacco.",
        ]

    if {"trestbps", "chol"} & set(top_risk_factors):
        actions.append("Prepare recent blood pressure and cholesterol measurements for review.")
    if {"cp", "exang", "thalach", "oldpeak"} & set(top_risk_factors):
        actions.append("Write down when symptoms occur, whether they appear during exertion, and how long they last.")
    if caffeine_alert in {"high", "watch", "moderate"}:
        actions.append("Bring today's caffeine and energy-drink log to the consultation as lifestyle context.")
    return actions


def build_consultation_questions(
    top_risk_factors: list[str],
    symptoms: dict[str, bool],
    confidence: str,
) -> list[str]:
    questions = [
        "Do my entered values suggest that I should repeat any measurements or get a formal checkup?",
        "Which risk factor should I prioritize first: blood pressure, cholesterol, blood sugar, symptoms, activity, sleep, or tobacco exposure?",
        "What symptoms would mean I should seek urgent care instead of waiting for a routine appointment?",
    ]
    if {"trestbps", "chol"} & set(top_risk_factors):
        questions.append("Are my blood pressure or cholesterol values high enough to require follow-up testing or management?")
    if {"cp", "exang", "thalach", "oldpeak"} & set(top_risk_factors):
        questions.append("Could my chest-pain or exercise-related symptoms require ECG, stress testing, or another evaluation?")
    if any(symptoms.values()):
        questions.append("Given my current warning symptoms, should I go to emergency care now?")
    if confidence == "low":
        questions.append("Which missing or uncertain values should I measure before relying on this risk summary?")
    questions.append("Could caffeine or energy-drink intake be contributing to palpitations, sleep loss, or symptom perception?")
    return questions


def build_visit_summary(result: dict, input_features) -> dict:
    frame = as_feature_frame(input_features)
    first_row = frame.iloc[0].to_dict() if not frame.empty else {}
    return {
        "purpose": "Consultation preparation summary for a clinician or campus health center.",
        "heart_probability_percent": result["heart_probability_percent"],
        "action_level": result["action_level"],
        "confidence_level": result["confidence_level"],
        "top_risk_factors": result["top_risk_factors"],
        "key_measurements": {
            key: first_row.get(key)
            for key in ["age", "trestbps", "chol", "thalach", "oldpeak", "cp", "exang"]
        },
        "daily_caffeine_mg": result["daily_caffeine_mg"],
        "caffeine_alert": result["caffeine_alert"],
        "disclaimer": result["medical_disclaimer"],
    }


def student_message(action_level: str, alert: str, confidence: str) -> str:
    if action_level == "urgent":
        return (
            "Warning symptoms were reported. This is not a diagnosis; seek urgent medical help "
            "through 119, an emergency department, or a qualified clinician."
        )
    if action_level == "consult":
        return (
            "The model shows a high heart-risk signal. Bring the visit summary to a campus "
            "health center or clinician for review."
        )
    if action_level == "watch":
        return "Review uncertain values and watch for warning symptoms; consider consultation if the signal repeats."
    if alert in {"high", "watch", "moderate"}:
        return "Heart-risk signal is low, but caffeine or sugar intake is notable today."
    if confidence == "low":
        return "Confidence is low because many clinical profile values are missing."
    return "No strong warning signal from the provided values. This is guidance, not diagnosis."


def predict(features, beverage_log=None, model_bundle: dict | None = None) -> dict:
    bundle = load_model_bundle() if model_bundle is None else model_bundle
    frame = validate_clinical_ranges(as_feature_frame(features), allow_missing=True)
    probabilities = bundle["pipeline"].predict_proba(frame)[:, 1]
    beverage_summary = summarize_beverage_log(beverage_log)
    alert = caffeine_alert(beverage_summary["daily_caffeine_mg"], beverage_summary["daily_sugar_g"])
    probability = float(probabilities[0])
    tier = risk_tier(probability)
    confidence = confidence_level(frame, probability)
    symptoms = extract_symptoms(features)
    action_level = determine_action_level(probability, symptoms, confidence)
    top_risk_factors = bundle.get("selected_original_features", [])[:5]

    result = {
        "heart_probability": round(probability, 4),
        "heart_probability_percent": round(probability * 100, 1),
        "risk_tier": tier,
        "action_level": action_level,
        "confidence_level": confidence,
        "top_risk_factors": top_risk_factors,
        "symptoms": symptoms,
        "daily_caffeine_mg": beverage_summary["daily_caffeine_mg"],
        "daily_calories_kcal": beverage_summary["daily_calories_kcal"],
        "daily_sugar_g": beverage_summary["daily_sugar_g"],
        "caffeine_alert": alert,
        "response_actions": build_response_actions(action_level, top_risk_factors, alert),
        "consultation_questions": build_consultation_questions(top_risk_factors, symptoms, confidence),
        "medical_disclaimer": MEDICAL_DISCLAIMER,
        "student_message": student_message(action_level, alert, confidence),
        "model_version": bundle.get("model_version", "unknown"),
        "batch_probabilities": [round(float(value), 4) for value in probabilities],
    }
    result["visit_summary"] = build_visit_summary(result, features)
    return result


def log_inference(result: dict, input_shape, actual=None, log_path: Path | None = None) -> None:
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = log_path or MLRUNS_DIR / "inference.log"
    logger = logging.getLogger("cardiocare.inference")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": result.get("model_version"),
        "input_shape": list(input_shape),
        "prediction": result.get("batch_probabilities"),
        "actual": actual,
    }
    logger.info(json.dumps(record, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CardioCare batch inference.")
    parser.add_argument("--input", default="data/sample_input.csv")
    parser.add_argument("--beverages", default="data/sample_beverage_log.csv")
    parser.add_argument("--output", default="mlruns/sample_prediction.json")
    args = parser.parse_args()

    features = pd.read_csv(args.input)
    beverage_log = pd.read_csv(args.beverages) if Path(args.beverages).exists() else None
    result = predict(features, beverage_log)
    log_inference(result, features.shape)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
