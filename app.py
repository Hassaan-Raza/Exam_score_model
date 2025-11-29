import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np

st.set_page_config(
    page_title="Exam Score Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model():
    return joblib.load("exam_score_model.pkl")


@st.cache_resource
def load_feature_info():
    with open("model_features.json", "r") as f:
        return json.load(f)


model = load_model()
feature_info = load_feature_info()
st.success("✅ Model loaded successfully!")

st.title("Student Exam Score Predictor")
st.markdown("Predict student exam scores based on study habits, lifestyle, and learning environment.")

# Create columns for better organization
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Personal Information")
    age = st.number_input("Age", min_value=15, max_value=30, value=20)
    gender = st.selectbox("Gender", ["male", "female"])
    course = st.selectbox("Course", ["btech", "bca", "mca", "mtech"])

with col2:
    st.header("Study Habits")
    study_hours = st.slider("Daily Study Hours", 0.0, 12.0, 3.0, 0.5)
    class_attendance = st.slider("Class Attendance (%)", 0, 100, 75)
    study_method = st.selectbox("Study Method", ["self-study", "coaching", "online videos", "group study"])

with col3:
    st.header("Lifestyle & Environment")
    sleep_hours = st.slider("Daily Sleep Hours", 0.0, 12.0, 7.0, 0.5)
    sleep_quality = st.selectbox("Sleep Quality", ["poor", "average", "good"])
    internet_access = st.selectbox("Internet Access", ["yes", "no"])
    facility_rating = st.selectbox("Facility Rating", ["low", "medium", "high"])

# Exam Information below the columns
st.header("Exam Information")
exam_difficulty = st.selectbox("Exam Difficulty", ["easy", "moderate", "hard"])

# Calculate engineered features (with safety checks)
study_attendance_ratio = study_hours / (class_attendance / 100) if class_attendance > 0 else 0
sleep_study_balance = sleep_hours / study_hours if study_hours > 0 else 0
efficiency_score = (class_attendance * study_hours) / 100
has_high_attendance = 1 if class_attendance >= 80 else 0
has_adequate_sleep = 1 if sleep_hours >= 8 else 0

# Sleep quality mapping
sleep_quality_map = {"poor": 0, "average": 1, "good": 2}

# Prepare input data - MATCH THE EXACT DATA TYPES FROM TRAINING
student_data = {
    'age': int(age),
    'gender': str(gender),
    'course': str(course),
    'study_hours': float(study_hours),
    'class_attendance': int(class_attendance),
    'internet_access': str(internet_access),
    'sleep_hours': float(sleep_hours),
    'sleep_quality': int(sleep_quality_map[sleep_quality]),  # Convert to int
    'study_method': str(study_method),
    'facility_rating': str(facility_rating),  # Keep as string for one-hot encoding
    'exam_difficulty': str(exam_difficulty),
    'study_attendance_ratio': float(study_attendance_ratio),
    'sleep_study_balance': float(sleep_study_balance),
    'efficiency_score': float(efficiency_score),
    'has_high_attendance': int(has_high_attendance),
    'has_adequate_sleep': int(has_adequate_sleep)
}

# Create DataFrame with exact column order from training
input_df = pd.DataFrame([student_data])

# Ensure column order matches training data
expected_columns = feature_info['all_features']
input_df = input_df.reindex(columns=expected_columns, fill_value=0)

# Center the predict button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🎯 Predict Exam Score", use_container_width=True)

if predict_button:
    with st.spinner("Analyzing student data..."):
        # Make prediction
        predicted_score = model.predict(input_df)[0]

        # Determine performance category
        if predicted_score >= 85:
            category = "EXCELLENT"
            color = "green"
            advice = "Excellent work! Maintain your current habits and continue striving for excellence."
        elif predicted_score >= 70:
            category = "GOOD"
            color = "blue"
            advice = "Good performance! Focus on consistent study habits and you can improve even further."
        elif predicted_score >= 60:
            category = "AVERAGE"
            color = "orange"
            advice = "Average performance. Consider improving study habits and attendance for better results."
        else:
            category = "NEEDS IMPROVEMENT"
            color = "red"
            advice = "Needs improvement. Focus on regular study, better attendance, and adequate sleep."

        # Display results
        st.success("Prediction Complete!")

        # Results in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Predicted Exam Score",
                value=f"{predicted_score:.1f}",
                delta="Score"
            )

        with col2:
            st.metric(
                label="Performance Category",
                value=category,
                delta=""
            )

        with col3:
            eff_label = "High" if efficiency_score > 2 else "Medium" if efficiency_score > 1 else "Low"
            st.metric(
                label="Study Efficiency",
                value=f"{efficiency_score:.1f}",
                delta=eff_label
            )