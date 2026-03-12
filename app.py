import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Disease Risk Predictor",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    .main {
        background-color: #f6f8fb;
    }

    .app-title {
        font-size: 42px;
        font-weight: 800;
        color: #12344d;
        margin-bottom: 0.2rem;
    }

    .app-subtitle {
        font-size: 18px;
        color: #5b6b7a;
        margin-bottom: 1.5rem;
    }

    .card {
        background-color: white;
        padding: 1.2rem 1.2rem 0.8rem 1.2rem;
        border-radius: 16px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
    }

    .result-high {
        background: #ffe3e3;
        color: #b42318;
        padding: 1rem;
        border-radius: 14px;
        font-size: 24px;
        font-weight: 700;
        text-align: center;
        border: 1px solid #f5b5b5;
    }

    .result-low {
        background: #dcfce7;
        color: #166534;
        padding: 1rem;
        border-radius: 14px;
        font-size: 24px;
        font-weight: 700;
        text-align: center;
        border: 1px solid #b7ebc6;
    }

    .small-note {
        color: #667085;
        font-size: 14px;
    }

    .stButton > button {
        width: 100%;
        height: 3.2rem;
        border-radius: 12px;
        font-size: 18px;
        font-weight: 700;
        background-color: #1565c0;
        color: white;
        border: none;
    }

    .stButton > button:hover {
        background-color: #0d47a1;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# HELPERS
# =========================================================
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "disease_risk_model.pkl"


@st.cache_resource
def load_model(model_path: Path):
    with open(model_path, "rb") as file:
        loaded_model = pickle.load(file)
    return loaded_model


def encode_binary(value: str) -> int:
    mapping = {
        "Yes": 1,
        "No": 0,
        "Male": 1,
        "Female": 0
    }
    return mapping.get(value, 0)


def build_input_dataframe(
    include_id: bool,
    patient_id: int,
    age: int,
    gender: str,
    bmi: float,
    daily_steps: int,
    sleep_hours: float,
    water_intake_l: float,
    calories_consumed: int,
    smoker: str,
    alcohol: str,
    resting_hr: int,
    systolic_bp: int,
    diastolic_bp: int,
    cholesterol: int,
    family_history: str
) -> pd.DataFrame:
    data = {
        "age": [age],
        "gender": [encode_binary(gender)],
        "bmi": [bmi],
        "daily_steps": [daily_steps],
        "sleep_hours": [sleep_hours],
        "water_intake_l": [water_intake_l],
        "calories_consumed": [calories_consumed],
        "smoker": [encode_binary(smoker)],
        "alcohol": [encode_binary(alcohol)],
        "resting_hr": [resting_hr],
        "systolic_bp": [systolic_bp],
        "diastolic_bp": [diastolic_bp],
        "cholesterol": [cholesterol],
        "family_history": [encode_binary(family_history)],
    }

    if include_id:
        data = {"id": [patient_id], **data}

    return pd.DataFrame(data)


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Settings")
st.sidebar.info(
    "Make sure `disease_risk_model.pkl` is in the same folder as `app.py`."
)

include_id = st.sidebar.checkbox(
    "Model was trained with 'id' column",
    value=False,
    help="Turn this on only if your training data included the 'id' column as a feature."
)

show_input_table = st.sidebar.checkbox("Show input data table", value=True)


# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="app-title">🩺 Disease Risk Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Enter health details below and predict disease risk using your trained Decision Tree model.</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================
if not MODEL_PATH.exists():
    st.error(
        f"Model file not found.\n\n"
        f"Expected location: `{MODEL_PATH}`\n\n"
        f"Put `disease_risk_model.pkl` in the same folder as `app.py`."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()


# =========================================================
# INPUT FORM
# =========================================================
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Patient Details")

    if include_id:
        patient_id = st.number_input("Patient ID", min_value=1, value=1, step=1)
    else:
        patient_id = 1

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.5, step=0.1)
        daily_steps = st.number_input("Daily Steps", min_value=0, max_value=50000, value=6000, step=100)

    with col2:
        sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.1)
        water_intake_l = st.number_input("Water Intake (L)", min_value=0.0, max_value=10.0, value=2.5, step=0.1)
        calories_consumed = st.number_input("Calories Consumed", min_value=0, max_value=10000, value=2200, step=50)
        smoker = st.selectbox("Smoker", ["No", "Yes"])

    with col3:
        alcohol = st.selectbox("Alcohol", ["No", "Yes"])
        resting_hr = st.number_input("Resting Heart Rate", min_value=30, max_value=200, value=72, step=1)
        systolic_bp = st.number_input("Systolic BP", min_value=50, max_value=250, value=120, step=1)
        diastolic_bp = st.number_input("Diastolic BP", min_value=30, max_value=150, value=80, step=1)

    col4, col5 = st.columns(2)
    with col4:
        cholesterol = st.number_input("Cholesterol", min_value=50, max_value=500, value=180, step=1)
    with col5:
        family_history = st.selectbox("Family History", ["No", "Yes"])

    predict_button = st.button("Predict Disease Risk")
    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# PREDICTION
# =========================================================
if predict_button:
    try:
        input_data = build_input_dataframe(
            include_id=include_id,
            patient_id=patient_id,
            age=age,
            gender=gender,
            bmi=bmi,
            daily_steps=daily_steps,
            sleep_hours=sleep_hours,
            water_intake_l=water_intake_l,
            calories_consumed=calories_consumed,
            smoker=smoker,
            alcohol=alcohol,
            resting_hr=resting_hr,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            cholesterol=cholesterol,
            family_history=family_history
        )

        prediction = model.predict(input_data)[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_data)[0]
            confidence = float(max(probs) * 100)

        st.subheader("Prediction Result")

        if str(prediction).lower() in ["1", "high", "high risk", "yes"]:
            st.markdown(
                '<div class="result-high">⚠️ High Disease Risk</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-low">✅ Low Disease Risk</div>',
                unsafe_allow_html=True
            )

        if confidence is not None:
            st.metric("Model Confidence", f"{confidence:.2f}%")

        if show_input_table:
            st.subheader("Processed Input Data")
            st.dataframe(input_data, use_container_width=True)

        csv_data = input_data.copy()
        csv_data["prediction"] = prediction
        if confidence is not None:
            csv_data["confidence_percent"] = round(confidence, 2)

        st.download_button(
            label="Download Prediction Result as CSV",
            data=csv_data.to_csv(index=False).encode("utf-8"),
            file_name="disease_risk_prediction.csv",
            mime="text/csv"
        )

        st.caption(
            "Note: This prediction depends on how the model was trained. "
            "The encoding of Gender, Smoker, Alcohol, and Family History must match the training process."
        )

    except ValueError as e:
        st.error(f"Prediction failed due to input mismatch: {e}")
        st.info(
            "This usually happens when the model expects different columns than the app is sending. "
            "Check whether your model was trained with the `id` column and whether categorical values were encoded the same way."
        )
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    "<div class='small-note'>Built with Streamlit • Decision Tree Classifier • Disease Risk Prediction</div>",
    unsafe_allow_html=True
)