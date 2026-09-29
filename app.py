import json
import time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# Load model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "pipeline_student_impact.joblib"
METADATA_PATH = BASE_DIR / "model_metadata.json"


@st.cache_resource
def load_model():
    pipeline = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return pipeline, metadata


pipeline, metadata = load_model()


# Prediction
def predict(features):
    input_df = pd.DataFrame([features])

    start = time.perf_counter()

    probabilities = pipeline.predict_proba(input_df)[0]
    prediction = int(pipeline.predict(input_df)[0])

    latency_ms = (time.perf_counter() - start) * 1000

    class_probabilities = {
        int(label): float(probability)
        for label, probability in zip(
            pipeline.classes_,
            probabilities
        )
    }

    confidence = max(class_probabilities.values())

    return prediction, confidence, class_probabilities, latency_ms


def check_input_range(features):
    warnings = []

    for feature, bounds in metadata["numeric_ranges"].items():
        value = float(features[feature])

        if value < bounds["min"] or value > bounds["max"]:
            warnings.append(feature)

    return warnings


# UI
st.title("AI Student Impact")

st.write("Predict whether GPA is likely to improve.")

with st.form("prediction_form"):

    traditional_hours = st.number_input(
        "Traditional Study Hours / Week",
        min_value=0.0,
        max_value=80.0,
        value=10.0
    )

    genai_hours = st.number_input(
        "Generative AI Hours / Week",
        min_value=0.0,
        max_value=80.0,
        value=5.0
    )

    year = st.selectbox(
        "Year of Study",
        metadata["categories"]["Year_of_Study"]
    )

    use_case = st.selectbox(
        "Primary AI Use Case",
        metadata["categories"]["Primary_Use_Case"]
    )

    skill = st.selectbox(
        "Prompt Engineering Skill",
        metadata["categories"]["Prompt_Engineering_Skill"]
    )

    dependency = st.slider(
        "AI Dependency",
        1,
        10,
        3
    )

    anxiety = st.slider(
        "Exam Anxiety",
        1,
        10,
        4
    )

    submitted = st.form_submit_button("Predict")


# Result
if submitted:

    features = {
        "Traditional_Study_Hours": traditional_hours,
        "Year_of_Study": year,
        "Primary_Use_Case": use_case,
        "Weekly_GenAI_Hours": genai_hours,
        "Prompt_Engineering_Skill": skill,
        "Perceived_AI_Dependency": dependency,
        "Anxiety_Level_During_Exams": anxiety,
    }

    prediction, confidence, probabilities, latency = predict(features)

    st.divider()
    st.subheader("Result")

    if prediction == 1:
        st.success("GPA likely to improve")
    else:
        st.error("GPA unlikely to improve")

    st.write(f"Confidence: {confidence:.1%}")

    st.write(
        f"GPA Improved: {probabilities.get(1, 0):.1%}"
    )

    st.write(
        f"GPA Not Improved: {probabilities.get(0, 0):.1%}"
    )

    st.caption(f"Prediction latency: {latency:.2f} ms")

    # Confidence threshold
    threshold = metadata["confidence_threshold"]

    if confidence < threshold:
        st.warning(
            f"Low confidence (< {threshold:.0%})"
        )

    # Out-of-range input
    out_of_range = check_input_range(features)

    if out_of_range:
        st.warning(
            "Input outside training range: "
            + ", ".join(out_of_range)
        )