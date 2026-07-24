import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# --- Custom CSS ---
st.markdown(
    """
    <style>
    /* Overall app background */
    .stApp {
        background: linear-gradient(180deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: #f5f5f5;
    }

    /* Title */
    h1 {
        color: #ffffff !important;
        text-align: center;
        font-weight: 800;
        padding-bottom: 0px;
    }

    /* Subheaders */
    h3 {
        color: #ffd369 !important;
    }

    /* Intro text */
    .stMarkdown p {
        color: #dfe6e9;
        text-align: center;
        font-size: 16px;
    }

    /* Divider color */
    hr {
        border-color: rgba(255,255,255,0.2) !important;
    }

    /* Card-like container for inputs */
    div[data-testid="column"] {
        background-color: rgba(255, 255, 255, 0.06);
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* Labels */
    label, .stSlider label, .stSelectbox label, .stNumberInput label {
        color: #f5f5f5 !important;
        font-weight: 600 !important;
    }

    /* Predict button */
    div.stButton > button {
        background: linear-gradient(90deg, #ff6b6b, #ffd369);
        color: #1a1a1a;
        font-weight: 700;
        font-size: 18px;
        border: none;
        border-radius: 12px;
        padding: 12px 0px;
        transition: 0.3s ease-in-out;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0px 4px 20px rgba(255, 211, 105, 0.5);
        color: #000000;
    }

    /* Metric styling */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 10px;
        text-align: center;
    }
    div[data-testid="stMetricValue"] {
        color: #ffd369 !important;
    }

    /* Success / Error boxes */
    div.stAlert {
        border-radius: 12px;
        font-size: 17px;
    }

    /* Caption footer */
    .stCaption {
        text-align: center;
        color: #b2bec3 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the saved model and expected columns
model = joblib.load('rf_model.pkl')
model_columns = joblib.load('model_columns.pkl')

# --- Header ---
st.title("🚢 Titanic Survival Predictor")
st.markdown(
    """
    Enter a passenger's details below and this model will estimate 
    their chances of surviving the Titanic disaster, based on patterns 
    learned from real historical passenger data.
    """
)
st.divider()

# --- Input form ---
st.subheader("👤 Passenger Details")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1 = Upper, 2 = Middle, 3 = Lower")
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 80, 30)

with col2:
    fare = st.number_input("Fare ($)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"], help="S = Southampton, C = Cherbourg, Q = Queenstown")
    family_size = st.slider("Family Size (including self)", 1, 11, 1)

is_alone = 1 if family_size == 1 else 0

st.divider()

# --- Prediction ---
if st.button("🔮 Predict Survival", use_container_width=True):
    input_dict = {
        'Pclass': pclass,
        'Age': age,
        'Fare': fare,
        'FamilySize': family_size,
        'IsAlone': is_alone,
        'Sex_male': 1 if sex == 'male' else 0,
        'Embarked_Q': 1 if embarked == 'Q' else 0,
        'Embarked_S': 1 if embarked == 'S' else 0,
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Result")

    if prediction == 1:
        st.success("✅ Predicted: **Survived**")
    else:
        st.error("❌ Predicted: **Did Not Survive**")

    st.metric(label="Survival Probability", value=f"{probability:.1%}")
    st.progress(probability)

# --- Footer ---
st.divider()
st.caption("Built as part of the Neurofive Solutions ML Internship Track · Model: Random Forest Classifier · Dataset: Titanic (Kaggle)")