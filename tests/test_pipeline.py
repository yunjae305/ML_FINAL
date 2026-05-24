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

from inference import (  # noqa: E402
    MEDICAL_DISCLAIMER,
    build_consultation_questions,
    build_visit_summary,
    determine_action_level,
)
from preprocessing import (  # noqa: E402
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    build_model_pipeline,
    load_beverages,
    load_heart_data,
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


class TestBeverageLayer(unittest.TestCase):
    def test_beverage_search_and_summary(self):
        beverages = load_beverages()
        matches = search_beverages("coffee", beverages)
        self.assertGreaterEqual(len(matches), 1)
        log = pd.DataFrame({"drink_id": [matches.iloc[0]["drink_id"]], "quantity": [2]})
        summary = summarize_beverage_log(log)
        self.assertGreater(summary["daily_caffeine_mg"], 0)


class TestActionNavigator(unittest.TestCase):
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
        }
        features = {column: 0 for column in FEATURE_COLUMNS}
        features.update({"age": 22, "trestbps": 120, "chol": 200, "thalach": 170, "oldpeak": 0.0})
        summary = build_visit_summary(result, features)
        self.assertEqual(summary["heart_probability_percent"], 72.0)
        self.assertEqual(summary["action_level"], "consult")
        self.assertIn("top_risk_factors", summary)
        self.assertIn("disclaimer", summary)


if __name__ == "__main__":
    unittest.main()
