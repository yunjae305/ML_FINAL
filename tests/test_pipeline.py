import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

import inference
import preprocessing
from inference import (
    MEDICAL_DISCLAIMER,
    build_consultation_questions,
    build_guidance_sources,
    build_visit_summary,
    determine_action_level,
    predict,
)
from monitor import make_drifted_test_set
from train import _artifact_location_points_to_current_mlruns
from train import clinical_selection_rationale
from preprocessing import (
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    _build_report_sections,
    build_model_pipeline,
    load_beverages,
    load_heart_data,
    prepare_model_data,
    search_beverages,
    summarize_beverage_log,
    validate_clinical_ranges,
)


class TestCardioCarePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = load_heart_data()
        X = df[FEATURE_COLUMNS]
        y = df[TARGET_COLUMN]
        X_train, X_test, y_train, _ = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        cls.X_sample = X_test.head(8).reset_index(drop=True)
        cls.pipeline = build_model_pipeline(
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="liblinear",
                random_state=RANDOM_STATE,
            )
        )
        cls.pipeline.fit(X_train, y_train)

    def test_prediction_shape_matches_input_shape(self):
        predictions = self.pipeline.predict(self.X_sample)
        self.assertEqual(predictions.shape[0], self.X_sample.shape[0])

    def test_predict_proba_is_valid_probability_distribution(self):
        probabilities = self.pipeline.predict_proba(self.X_sample)
        self.assertEqual(probabilities.shape, (self.X_sample.shape[0], 2))
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))
        np.testing.assert_allclose(
            probabilities.sum(axis=1),
            np.ones(self.X_sample.shape[0]),
            atol=1e-6,
        )

    def test_cholesterol_range_validation(self):
        invalid = self.X_sample.copy()
        invalid.loc[0, "chol"] = 999
        with self.assertRaises(ValueError):
            validate_clinical_ranges(invalid)

    def test_pipeline_is_deterministic_for_same_input(self):
        first = self.pipeline.predict_proba(self.X_sample)
        second = self.pipeline.predict_proba(self.X_sample)
        np.testing.assert_allclose(first, second, atol=1e-12)

    def test_prepare_model_data_removes_duplicates_and_empty_columns(self):
        df = load_heart_data().head(3)
        duplicated = pd.concat([df, df.head(1)], ignore_index=True)
        duplicated["empty_column"] = np.nan
        cleaned = prepare_model_data(duplicated)
        self.assertEqual(cleaned.shape[0], 3)
        self.assertNotIn("empty_column", cleaned.columns)
        self.assertEqual(cleaned.columns.tolist(), FEATURE_COLUMNS + [TARGET_COLUMN])

    def test_pipeline_uses_single_job_for_unicode_workspace_paths(self):
        selector = self.pipeline.named_steps["selector"]
        self.assertEqual(selector.estimator.n_jobs, 1)


class TestBeverageLayer(unittest.TestCase):
    def test_beverage_search_and_summary(self):
        beverages = load_beverages()
        matches = search_beverages("coffee", beverages)
        self.assertGreaterEqual(len(matches), 1)
        log = pd.DataFrame({"drink_id": [matches.iloc[0]["drink_id"]], "quantity": [2]})
        summary = summarize_beverage_log(log)
        self.assertGreater(summary["daily_caffeine_mg"], 0)


class TestActionNavigator(unittest.TestCase):
    def test_medical_disclaimer_warns_not_professional_diagnosis(self):
        self.assertIn("전문적인 의학 진단", MEDICAL_DISCLAIMER)
        self.assertIn("qualified clinician", MEDICAL_DISCLAIMER)

    def test_high_probability_routes_to_consult(self):
        symptoms = {
            "symptom_chest_pain_now": False,
            "symptom_shortness_breath": False,
            "symptom_fainting": False,
            "symptom_radiating_pain": False,
        }
        self.assertEqual(determine_action_level(0.72, symptoms, "high"), "consult")

    def test_symptom_routes_to_urgent(self):
        symptoms = {
            "symptom_chest_pain_now": True,
            "symptom_shortness_breath": False,
            "symptom_fainting": False,
            "symptom_radiating_pain": False,
        }
        self.assertEqual(determine_action_level(0.10, symptoms, "high"), "urgent")

    def test_consultation_questions_are_nonempty(self):
        questions = build_consultation_questions(
            ["trestbps", "chol", "cp"],
            {"symptom_chest_pain_now": False},
            "medium",
        )
        self.assertGreaterEqual(len(questions), 3)

    def test_visit_summary_contains_required_fields(self):
        result = {
            "heart_probability_percent": 72.0,
            "action_level": "consult",
            "confidence_level": "high",
            "top_risk_factors": ["trestbps", "chol"],
            "daily_caffeine_mg": 120.0,
            "caffeine_alert": "low",
            "medical_disclaimer": MEDICAL_DISCLAIMER,
            "risk_summary_svg": "<svg></svg>",
        }
        features = {column: 0 for column in FEATURE_COLUMNS}
        features.update({"age": 22, "trestbps": 120, "chol": 200, "thalach": 170, "oldpeak": 0.0})
        summary = build_visit_summary(result, features)
        self.assertEqual(summary["heart_probability_percent"], 72.0)
        self.assertEqual(summary["action_level"], "consult")
        self.assertIn("top_risk_factors", summary)
        self.assertIn("disclaimer", summary)
        self.assertIn("risk_summary_svg", summary)

    def test_guidance_sources_are_brief_institution_labels(self):
        sources = build_guidance_sources("urgent", "moderate")
        self.assertIn("출처: CDC", sources)
        self.assertIn("출처: American Heart Association", sources)
        self.assertIn("출처: FDA", sources)
        self.assertIn("출처: MedlinePlus", sources)
        self.assertTrue(all(source.startswith("출처: ") for source in sources))
        self.assertTrue(all("http" not in source for source in sources))

    def test_prediction_result_includes_brief_guidance_sources(self):
        class FakePipeline:
            def predict_proba(self, frame):
                return np.array([[0.2, 0.8]])

        bundle = {
            "pipeline": FakePipeline(),
            "model_version": "test",
            "selected_original_features": ["trestbps", "chol"],
        }
        features = {column: 0 for column in FEATURE_COLUMNS}
        features.update({"age": 22, "trestbps": 120, "chol": 190, "thalach": 175, "oldpeak": 0.1})
        result = predict(features, model_bundle=bundle)
        self.assertIn("guidance_sources", result)
        self.assertIn("출처: CDC", result["guidance_sources"])
        self.assertTrue(all("http" not in source for source in result["guidance_sources"]))

    def test_risk_summary_svg_contains_probability_action_and_confidence(self):
        self.assertTrue(hasattr(inference, "build_risk_summary_svg"))
        svg = inference.build_risk_summary_svg(72.0, "consult", "medium", "watch")
        self.assertIn("<svg", svg)
        self.assertIn("72.0%", svg)
        self.assertIn("consult", svg)
        self.assertIn("medium", svg)

    def test_prediction_result_includes_svg_risk_summary(self):
        class FakePipeline:
            def predict_proba(self, frame):
                return np.array([[0.28, 0.72]])

        bundle = {
            "pipeline": FakePipeline(),
            "model_version": "test",
            "selected_original_features": ["trestbps", "chol"],
        }
        features = {column: 0 for column in FEATURE_COLUMNS}
        features.update({"age": 22, "trestbps": 120, "chol": 190, "thalach": 175, "oldpeak": 0.1})
        result = predict(features, model_bundle=bundle)
        self.assertIn("risk_summary_svg", result)
        self.assertIn("<svg", result["risk_summary_svg"])
        self.assertIn("72.0%", result["risk_summary_svg"])
        self.assertIn("risk_summary_svg", result["visit_summary"])


class TestFinalReportRequirements(unittest.TestCase):
    def test_stale_mlflow_artifact_location_is_not_current(self):
        stale = "file:C:\\Users\\kyj31\\Desktop\\ML_Final\\mlruns/779756512248946338"
        current = str(ROOT_DIR / "mlruns")
        malformed_child = (ROOT_DIR / "mlruns" / "reproducible_1").as_uri()
        self.assertFalse(_artifact_location_points_to_current_mlruns(stale))
        self.assertFalse(_artifact_location_points_to_current_mlruns(malformed_child))
        self.assertTrue(_artifact_location_points_to_current_mlruns(current))

    def test_report_sections_cover_serving_retraining_ethics_and_ai_disclosure(self):
        training = {
            "base_model_results": [
                {
                    "name": "Logistic Regression",
                    "metrics": {
                        "balanced_accuracy": 0.9,
                        "precision": 0.8,
                        "recall": 0.95,
                        "f1": 0.87,
                    },
                    "confusion_matrix": [[28, 5], [1, 27]],
                }
            ],
            "final_model": {
                "threshold": 0.48,
                "metrics": {
                    "balanced_accuracy": 0.9,
                    "precision": 0.8,
                    "recall": 0.95,
                    "f1": 0.87,
                },
                "confusion_matrix": [[28, 5], [1, 27]],
                "selected_original_features": ["age", "chol"],
            },
        }
        monitoring = {
            "baseline_balanced_accuracy": 0.9,
            "shifted_balanced_accuracy": 0.84,
            "balanced_accuracy_delta": -0.06,
            "drift_report": [
                {"feature": "chol", "ks_statistic": 0.35, "p_value": 0.001, "drift_flag": True}
            ],
        }
        text = "\n".join(
            [title for title, _ in _build_report_sections(training, monitoring)]
            + [line for _, lines in _build_report_sections(training, monitoring) for line in lines]
        )
        self.assertIn("전문적인 의학 진단", text)
        self.assertIn("서빙 선택", text)
        self.assertIn("재학습", text)
        self.assertIn("Human-in-the-loop", text)
        self.assertIn("폭주", text)
        self.assertIn("AI 도구", text)
        self.assertIn("그림 1 캡션", text)
        self.assertIn("결측값은 삭제 대신", text)
        self.assertIn("SVC는 balanced accuracy가 더 높지만", text)
        self.assertIn("actual label", text)
        self.assertIn("confusion_matrix.json", text)
        self.assertIn("selected_features.json", text)
        self.assertIn("model_family tag", text)
        self.assertIn("임상적으로 해석 가능한 특성", text)

    def test_report_metric_svg_contains_model_and_drift_information(self):
        training = {
            "final_model": {
                "threshold": 0.48,
                "metrics": {
                    "balanced_accuracy": 0.9,
                    "recall": 0.95,
                    "f1": 0.87,
                },
            },
        }
        monitoring = {"balanced_accuracy_delta": -0.06}
        self.assertTrue(hasattr(preprocessing, "_report_metric_svg"))
        svg = preprocessing._report_metric_svg(training, monitoring)
        self.assertIn("<svg", svg)
        self.assertIn("Balanced Accuracy", svg)
        self.assertIn("Recall", svg)
        self.assertIn("Drift delta", svg)


class TestCodeSubmissionRequirements(unittest.TestCase):
    def test_clinical_selection_rationale_has_required_depth(self):
        final_metrics = {"balanced_accuracy": 0.906, "precision": 0.844, "recall": 0.964, "f1": 0.9}
        final_cm = [[28, 5], [1, 27]]
        rationale = clinical_selection_rationale(
            "Logistic Regression threshold tuned",
            final_metrics,
            final_cm,
            [{"name": "SVC", "metrics": {"balanced_accuracy": 0.937, "recall": 0.964}}],
        )
        self.assertGreaterEqual(len(rationale), 3)
        self.assertLessEqual(len(rationale), 5)
        self.assertTrue(any("false negative" in sentence.lower() for sentence in rationale))

    def test_drift_helper_moves_mean_and_variance(self):
        test_set = load_heart_data().head(20)
        shifted = make_drifted_test_set(test_set)
        self.assertGreater(shifted["chol"].mean() - test_set["chol"].mean(), 30)
        self.assertGreater(shifted["chol"].std(), test_set["chol"].std())
        self.assertLess(shifted["thalach"].mean(), test_set["thalach"].mean())

    def test_dockerfile_copies_only_runtime_needed_paths(self):
        dockerfile = (ROOT_DIR / "Dockerfile").read_text(encoding="utf-8")
        self.assertNotIn("COPY . .", dockerfile)
        self.assertIn("COPY src ./src", dockerfile)
        self.assertIn("COPY mlruns/cardiocare_model.joblib ./mlruns/cardiocare_model.joblib", dockerfile)
        self.assertIn("src/inference.py", dockerfile)

    def test_ci_verifies_docker_build_and_run(self):
        workflow = (ROOT_DIR / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertIn("python src/monitor.py", workflow)
        self.assertIn("docker build -t cardiocare:1.0 .", workflow)
        self.assertIn("docker run --rm cardiocare:1.0", workflow)

    def test_readme_frames_project_as_cli_docker_ml_system(self):
        readme = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
        self.assertIn("CLI/Docker 기반 ML 시스템", readme)
        self.assertIn("전체 재현 절차", readme)
        self.assertIn("python src/monitor.py", readme)
        self.assertLess(readme.index("python src/monitor.py"), readme.index("docker build -t cardiocare:1.0 ."))
        self.assertIn("confusion_matrix.json", readme)
        self.assertIn("selected_features.json", readme)
        self.assertIn("model_family tag", readme)
        self.assertIn("monitoring_summary.json", readme)
        self.assertIn("전문적인 의학 진단", readme)

    def test_dockerignore_keeps_build_context_small(self):
        dockerignore = (ROOT_DIR / ".dockerignore").read_text(encoding="utf-8")
        patterns = set(dockerignore.splitlines())
        self.assertIn(".venv/", patterns)
        self.assertIn("mlruns/**", patterns)
        self.assertIn("!mlruns/", patterns)
        self.assertIn("!mlruns/cardiocare_model.joblib", patterns)
        self.assertIn("notebooks/", patterns)
        self.assertIn("report.pdf", patterns)

    def test_mlflow_model_logging_uses_current_name_argument(self):
        train_source = (ROOT_DIR / "src" / "train.py").read_text(encoding="utf-8")
        self.assertNotIn('artifact_path="model"', train_source)
        self.assertIn('name="model"', train_source)


if __name__ == "__main__":
    unittest.main()
