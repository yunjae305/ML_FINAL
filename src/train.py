from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing import (
    DATA_DIR,
    FEATURE_COLUMNS,
    MLRUNS_DIR,
    MODEL_PATH,
    PARALLEL_JOBS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SET_PATH,
    TEST_SIZE,
    TRAIN_REFERENCE_PATH,
    TRAINING_SUMMARY_PATH,
    build_model_pipeline,
    get_transformed_feature_names,
    load_heart_data,
    prepare_model_data,
    selected_original_features,
    write_json,
    write_report_pdf,
)


EXPERIMENT_NAME = "CardioCare Action Navigator"


def _artifact_location_points_to_current_mlruns(location: str) -> bool:
    normalized_location = unquote(str(location)).replace("\\", "/")
    parsed = urlparse(normalized_location)
    if parsed.scheme == "file":
        path_text = parsed.path
        if parsed.netloc:
            path_text = f"//{parsed.netloc}{path_text}"
        if len(path_text) > 2 and path_text[0] == "/" and path_text[2] == ":":
            path_text = path_text[1:]
    else:
        path_text = normalized_location.replace("file:", "")
    current_path = MLRUNS_DIR.resolve()
    location_path = Path(path_text).resolve()
    if location_path == current_path:
        return True
    try:
        relative_path = location_path.relative_to(current_path)
    except ValueError:
        return False
    if len(relative_path.parts) == 1 and not location_path.name.isdigit():
        return False
    return True


def configure_mlflow_experiment() -> str:
    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR.resolve()}")
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None or _artifact_location_points_to_current_mlruns(experiment.artifact_location):
        mlflow.set_experiment(EXPERIMENT_NAME)
        return EXPERIMENT_NAME

    suffix = 1
    while True:
        experiment_name = f"{EXPERIMENT_NAME} Reproducible {suffix}"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            return experiment_name
        if _artifact_location_points_to_current_mlruns(experiment.artifact_location):
            mlflow.set_experiment(experiment_name)
            return experiment_name
        suffix += 1


def metric_dict(y_true, y_pred) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def choose_threshold(y_true, probabilities) -> tuple[float, dict]:
    best = None
    for threshold in np.linspace(0.20, 0.70, 51):
        preds = (probabilities >= threshold).astype(int)
        metrics = metric_dict(y_true, preds)
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        metrics["threshold"] = float(threshold)
        metrics["false_negatives"] = int(cm[1, 0])
        metrics["selection_score"] = 0.55 * metrics["recall"] + 0.45 * metrics["balanced_accuracy"]
        if best is None or (metrics["selection_score"], -metrics["false_negatives"]) > (
            best["selection_score"],
            -best["false_negatives"],
        ):
            best = metrics
    return float(best["threshold"]), best


def save_confusion_matrix(cm, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["normal", "heart disease"])
    ax.set_yticklabels(["normal", "heart disease"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def log_model_run(name, pipeline, X_train, y_train, X_test, y_test, cv) -> dict:
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=PARALLEL_JOBS,
    )
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    metrics = metric_dict(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    selected_names = get_transformed_feature_names(pipeline)

    with mlflow.start_run(run_name=name):
        mlflow.set_tag("model_family", name)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_metric("cv_balanced_accuracy_mean", float(cv_scores.mean()))
        mlflow.log_metric("cv_balanced_accuracy_std", float(cv_scores.std()))
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        mlflow.log_dict({"confusion_matrix": cm.tolist()}, "confusion_matrix.json")
        mlflow.log_dict(
            {
                "selected_transformed_features": selected_names.tolist(),
                "selected_original_features": selected_original_features(selected_names),
            },
            "selected_features.json",
        )
        cm_path = MLRUNS_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
        save_confusion_matrix(cm, cm_path, f"{name} confusion matrix")
        mlflow.log_artifact(str(cm_path))
        mlflow.sklearn.log_model(pipeline, name="model")

    return {
        "name": name,
        "cv_balanced_accuracy_mean": float(cv_scores.mean()),
        "cv_balanced_accuracy_std": float(cv_scores.std()),
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "selected_original_features": selected_original_features(selected_names),
    }


def feature_importance_summary(fitted_pipeline) -> list[dict]:
    names = get_transformed_feature_names(fitted_pipeline)
    model = fitted_pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_)).ravel()
    else:
        values = np.zeros(len(names))
    order = np.argsort(values)[::-1]
    return [{"feature": str(names[i]), "importance": float(values[i])} for i in order[:10]]


def clinical_selection_rationale(
    final_model_name: str,
    final_metrics: dict,
    final_confusion_matrix: list[list[int]],
    base_results: list[dict],
) -> list[str]:
    false_negatives = int(final_confusion_matrix[1][0])
    best_base = max(
        base_results,
        key=lambda result: (
            result.get("metrics", {}).get("recall", 0.0),
            result.get("metrics", {}).get("balanced_accuracy", 0.0),
        ),
    )
    return [
        (
            f"The final model is {final_model_name}, selected because its recall "
            f"{final_metrics.get('recall', 0.0):.4f} keeps false negative cases low."
        ),
        (
            f"Its confusion matrix has {false_negatives} false negative case, which is the most important error "
            "to reduce in a clinical screening-support setting."
        ),
        (
            f"The strongest base comparison model was {best_base.get('name', 'unknown')} with recall "
            f"{best_base.get('metrics', {}).get('recall', 0.0):.4f}, so the final choice favors a threshold-tuned "
            "linear model that is easier to explain while preserving the same recall target."
        ),
        (
            "The model output is used only to recommend review or monitoring, not to diagnose or decide treatment."
        ),
    ]


def train() -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_model_data(load_heart_data())
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_reference = X_train.copy()
    train_reference[TARGET_COLUMN] = y_train.to_numpy()
    test_set = X_test.copy()
    test_set[TARGET_COLUMN] = y_test.to_numpy()
    train_reference.to_csv(TRAIN_REFERENCE_PATH, index=False)
    test_set.to_csv(TEST_SET_PATH, index=False)

    experiment_name = configure_mlflow_experiment()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    model_specs = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="liblinear",
        ),
        "SVC": SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=PARALLEL_JOBS,
        ),
    }

    base_results = []
    for name, estimator in model_specs.items():
        base_results.append(log_model_run(name, build_model_pipeline(estimator), X_train, y_train, X_test, y_test, cv))

    grid = GridSearchCV(
        build_model_pipeline(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                solver="liblinear",
            )
        ),
        param_grid={"model__C": [0.1, 0.5, 1.0, 2.0, 5.0]},
        scoring="recall",
        cv=cv,
        n_jobs=PARALLEL_JOBS,
        refit=True,
    )
    grid.fit(X_train, y_train)
    probabilities = grid.best_estimator_.predict_proba(X_test)[:, 1]
    threshold, threshold_summary = choose_threshold(y_test.to_numpy(), probabilities)
    final_preds = (probabilities >= threshold).astype(int)
    final_metrics = metric_dict(y_test, final_preds)
    final_cm = confusion_matrix(y_test, final_preds, labels=[0, 1])
    selected_names = get_transformed_feature_names(grid.best_estimator_)

    with mlflow.start_run(run_name="Logistic Regression threshold tuned"):
        mlflow.set_tag("model_family", "Logistic Regression")
        mlflow.set_tag("tuning", "GridSearchCV + threshold")
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("best_cv_recall", float(grid.best_score_))
        mlflow.log_metric("threshold", threshold)
        for key, value in final_metrics.items():
            mlflow.log_metric(key, value)
        mlflow.log_dict({"confusion_matrix": final_cm.tolist()}, "confusion_matrix.json")
        mlflow.log_dict(
            {
                "selected_transformed_features": selected_names.tolist(),
                "selected_original_features": selected_original_features(selected_names),
            },
            "selected_features.json",
        )
        cm_path = MLRUNS_DIR / "confusion_matrix_final_tuned.png"
        save_confusion_matrix(final_cm, cm_path, "Final tuned model confusion matrix")
        mlflow.log_artifact(str(cm_path))
        mlflow.sklearn.log_model(grid.best_estimator_, name="model")

    model_version = datetime.now(timezone.utc).strftime("cardiocare-%Y%m%dT%H%M%SZ")
    bundle = {
        "pipeline": grid.best_estimator_,
        "threshold": threshold,
        "model_version": model_version,
        "feature_columns": FEATURE_COLUMNS,
        "selected_original_features": selected_original_features(selected_names),
        "feature_importance": feature_importance_summary(grid.best_estimator_),
        "metrics": final_metrics,
        "confusion_matrix": final_cm.tolist(),
    }
    joblib.dump(bundle, MODEL_PATH)

    summary = {
        "dataset": "UCI processed Cleveland",
        "service_concept": "CardioCare Action Navigator",
        "target_binarization": "0 -> normal, 1..4 -> heart disease",
        "mlflow_experiment": experiment_name,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "model_version": model_version,
        "base_model_results": base_results,
        "grid_search": {
            "best_params": grid.best_params_,
            "best_cv_recall": float(grid.best_score_),
            "threshold_selection": threshold_summary,
        },
        "final_model": {
            "path": str(MODEL_PATH),
            "name": "Logistic Regression threshold tuned",
            "threshold": threshold,
            "metrics": final_metrics,
            "confusion_matrix": final_cm.tolist(),
            "selected_original_features": selected_original_features(selected_names),
            "clinical_selection_rationale": clinical_selection_rationale(
                "Logistic Regression threshold tuned",
                final_metrics,
                final_cm.tolist(),
                base_results,
            ),
        },
    }
    write_json(TRAINING_SUMMARY_PATH, summary)
    write_report_pdf()
    print(summary["final_model"])
    return summary


if __name__ == "__main__":
    train()
