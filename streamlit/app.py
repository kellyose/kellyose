import streamlit as st
import pandas as pd
import pickle
import os

# Configure the Streamlit app
st.set_page_config(page_title="Stroke Prediction App", page_icon="🧠", layout="centered")
st.markdown(
    """
    <style>
    .main { background-color: #f5f5f5; }
    h1 { color: #4CAF50; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 Stroke Prediction App")
st.subheader("Predict the likelihood of stroke based on patient details.")
st.markdown("Provide the following details in the sidebar to get a prediction.")

# Try to import imblearn and catch potential errors
try:
    import imblearn
except ImportError:
    st.error(
        "❌ The required library `imblearn` is not installed. "
        "Please add `imbalanced-learn` to your `requirements.txt` file and redeploy."
    )
    st.stop()

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), "lda_stroke_model.pkl")
        with open(model_path, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("❌ Model file not found! Ensure `lda_stroke_model.pkl` is in the same directory as `app.py`.")
        raise
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        raise

# Load the model
try:
    model = load_model()
except Exception as e:
    st.stop()

# Sidebar for user inputs
st.sidebar.title("Patient Details")
st.sidebar.info("Fill in the patient's details below:")

# Collect user inputs in the sidebar
with st.sidebar:
    age = st.number_input("🧓 Age (in years)", min_value=1, max_value=120, step=1)
    gender = st.radio("⚥ Gender", ["Male", "Female"])
    hypertension = st.radio("💊 Hypertension", ["No", "Yes"])
    heart_disease = st.radio("❤️ Heart Disease", ["No", "Yes"])
    avg_glucose_level = st.slider(
        "🩸 Average Glucose Level (mg/dL)", min_value=0.0, max_value=300.0, step=0.1
    )
    bmi = st.slider("⚖️ BMI (Body Mass Index)", min_value=0.0, max_value=50.0, step=0.1)

    # Additional Features
    smoking_status = st.selectbox(
        "🚬 Smoking Status",
        ["never smoked", "formerly smoked", "smokes", "Unknown"],
    )
    Residence_type = st.selectbox(
        "🏡 Residence Type", ["Urban", "Rural"]
    )
    work_type = st.selectbox(
        "💼 Work Type",
        ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"],
    )
    ever_married = st.radio("💍 Ever Married", ["No", "Yes"])

# Map user inputs to a DataFrame
data = {
    "age": age,
    "gender": 1 if gender == "Male" else 0,  # Adjust encoding if necessary
    "hypertension": 1 if hypertension == "Yes" else 0,
    "heart_disease": 1 if heart_disease == "Yes" else 0,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "smoking_status": smoking_status,  # Handle if needed in preprocessing
    "Residence_type": Residence_type,  # Handle if needed in preprocessing
    "work_type": work_type,  # Handle if needed in preprocessing
    "ever_married": ever_married,  # Handle if needed in preprocessing
}

input_df = pd.DataFrame([data])

# Main Section
st.header("Prediction Results")

if st.button("🔍 Predict"):
    try:
        # Perform prediction
        prediction = model.predict(input_df)
        result = "🟢 No Stroke" if prediction[0] == 0 else "🔴 Stroke"
        st.success(f"The predicted result is: **{result}**")
    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {str(e)}")

# Additional Information
st.markdown("---")
st.markdown(
    """
    ### How Does It Work?
    This app uses a machine learning model to predict the likelihood of a stroke based on:
    - **Age**
    - **Health history** (hypertension, heart disease, BMI)
    - **Lifestyle factors** (smoking, work type, residence type)
    
    ⚠️ **Disclaimer:** This is a predictive tool and not a substitute for medical advice. Consult a healthcare provider for accurate diagnosis and treatment.
    """
)

