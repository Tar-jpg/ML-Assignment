"""Train and export the HW6 production pipeline.

Run this script after placing ai_student_impact_dataset.csv beside it.
It reproduces the HW5 split/model-selection logic, exports the selected pipeline,
and writes metadata used by the Streamlit UI and monitoring demo.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "ai_student_impact_dataset.csv"
MODEL_PATH = BASE_DIR / "pipeline_student_impact.joblib"
METADATA_PATH = BASE_DIR / "model_metadata.json"
RANDOM_STATE = 42
CONFIDENCE_THRESHOLD = 0.65

FEATURES = [
    "Traditional_Study_Hours",
    "Year_of_Study",
    "Primary_Use_Case",
    "Weekly_GenAI_Hours",
    "Prompt_Engineering_Skill",
    "Perceived_AI_Dependency",
    "Anxiety_Level_During_Exams",
]

NUMERIC_FEATURES = [
    "Traditional_Study_Hours",
    "Weekly_GenAI_Hours",
    "Perceived_AI_Dependency",
    "Anxiety_Level_During_Exams",
]

YEAR_ORDER = ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"]
SKILL_ORDER = ["Beginner", "Intermediate", "Advanced"]


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", RobustScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                ["Primary_Use_Case"],
            ),
            (
                "ord",
                OrdinalEncoder(
                    categories=[YEAR_ORDER, SKILL_ORDER],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ["Year_of_Study", "Prompt_Engineering_Skill"],
            ),
        ]
    )


def json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def row_to_case(row: pd.Series, actual: int, probability: float) -> dict:
    values = {feature: json_value(row[feature]) for feature in FEATURES}
    prediction = int(probability >= 0.5)
    return {
        "features": values,
        "actual": int(actual),
        "prediction": prediction,
        "probability_improved": round(float(probability), 6),
        "confidence": round(float(max(probability, 1 - probability)), 6),
    }


def train_and_export() -> dict:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH.name}. Place the dataset beside train_model.py."
        )

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].copy()
    y = (df["Post_Semester_GPA"] > df["Pre_Semester_GPA"]).astype(int)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.15 / 0.85,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            C=1,
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    validation_results = []
    for name, estimator in models.items():
        candidate = Pipeline(
            [("prepare", make_preprocessor()), ("model", estimator)]
        )
        candidate.fit(X_train, y_train)
        prediction = candidate.predict(X_val)
        validation_results.append(
            {
                "model": name,
                "accuracy": float(accuracy_score(y_val, prediction)),
                "macro_f1": float(f1_score(y_val, prediction, average="macro")),
            }
        )

    validation_results.sort(key=lambda item: item["macro_f1"], reverse=True)
    best_model_name = validation_results[0]["model"]
    final_pipeline = Pipeline(
        [("prepare", make_preprocessor()), ("model", models[best_model_name])]
    )
    final_pipeline.fit(X_train_val, y_train_val)

    test_prediction = final_pipeline.predict(X_test)
    test_probability = final_pipeline.predict_proba(X_test)[:, 1]
    test_accuracy = float(accuracy_score(y_test, test_prediction))
    test_macro_f1 = float(f1_score(y_test, test_prediction, average="macro"))

    uncertainty = np.abs(test_probability - 0.5)
    low_confidence_position = int(np.argmin(uncertainty))

    correct_positions = np.flatnonzero(test_prediction == y_test.to_numpy())
    correct_confidences = np.maximum(
        test_probability[correct_positions], 1 - test_probability[correct_positions]
    )
    typical_position = int(correct_positions[np.argmax(correct_confidences)])

    joblib.dump(final_pipeline, MODEL_PATH)
    model_sha256 = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()

    numeric_ranges = {
        column: {
            "min": float(X_train_val[column].min()),
            "max": float(X_train_val[column].max()),
            "median": float(X_train_val[column].median()),
        }
        for column in NUMERIC_FEATURES
    }
    categories = {
        "Year_of_Study": YEAR_ORDER,
        "Primary_Use_Case": sorted(
            X_train_val["Primary_Use_Case"].dropna().unique().tolist()
        ),
        "Prompt_Engineering_Skill": SKILL_ORDER,
    }

    metadata = {
        "assignment": "HW6 Model Deployment",
        "model_version": "hw6-2026-09-28",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_sha256": model_sha256,
        "model_name": best_model_name,
        "dataset_rows": int(len(df)),
        "feature_order": FEATURES,
        "label_definition": "1 when Post_Semester_GPA > Pre_Semester_GPA; otherwise 0",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "random_state": RANDOM_STATE,
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "validation_results": validation_results,
        "test_metrics": {
            "accuracy": test_accuracy,
            "macro_f1": test_macro_f1,
        },
        "numeric_ranges": numeric_ranges,
        "categories": categories,
        "demo_cases": {
            "typical": row_to_case(
                X_test.iloc[typical_position],
                int(y_test.iloc[typical_position]),
                float(test_probability[typical_position]),
            ),
            "low_confidence": row_to_case(
                X_test.iloc[low_confidence_position],
                int(y_test.iloc[low_confidence_position]),
                float(test_probability[low_confidence_position]),
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "numpy": np.__version__,
        },
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


if __name__ == "__main__":
    result = train_and_export()
    print(f"Selected model: {result['model_name']}")
    print(f"Test accuracy: {result['test_metrics']['accuracy']:.4f}")
    print(f"Test Macro F1: {result['test_metrics']['macro_f1']:.4f}")
    print(f"Saved: {MODEL_PATH.name}")
    print(f"Saved: {METADATA_PATH.name}")
