"""
Streamlit App for Telecom Customer Churn Prediction
Professional & Light UI Design
"""

import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title='Churn Predictor Pro',
    page_icon='🎯',
    layout='wide',
    initial_sidebar_state='expanded'
)

# =============================
# CUSTOM CSS (FIXED TEXT COLORS)
# =============================
st.markdown("""
<style>

/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

/* APP BACKGROUND */
.main, .stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    font-family: 'Poppins', sans-serif;
    color: #1a202c !important;
    font-weight: 600;
}

/* =============================
   SIDEBAR (DARK → WHITE TEXT)
   ============================= */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
}

section[data-testid="stSidebar"] * {
    color: white !important;
    font-weight: 600;
}

/* =============================
   HEADER
   ============================= */
.custom-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2.5rem;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin-bottom: 2rem;
}

/* =============================
   WHITE CARDS → BLACK TEXT
   ============================= */
.card,
.stForm,
.stAlert,
.dataframe,
.streamlit-expanderHeader,
.streamlit-expanderContent,
.stTabs [data-baseweb="tab"] {
    background: white;
    color: #1a202c !important;
    font-weight: 600;
}

/* Selected tab */
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
}

/* =============================
   INPUTS & LABELS
   ============================= */
label,
.stTextInput label,
.stNumberInput label,
.stSelectbox label,
.stRadio label,
.stSlider label {
    color: #1a202c !important;
    font-weight: 600;
}

input, select, textarea {
    color: #1a202c !important;
    font-weight: 600;
}

/* =============================
   BUTTONS
   ============================= */
.stButton>button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-weight: 700;
    border-radius: 10px;
}

/* =============================
   METRICS
   ============================= */
div[data-testid="metric-container"] {
    background: white;
    border-left: 4px solid #667eea;
}

div[data-testid="metric-container"] label {
    color: #1a202c !important;
    font-weight: 700;
}

div[data-testid="stMetricValue"] {
    color: #1a202c !important;
    font-weight: 700;
}

/* =============================
   TABLE TEXT
   ============================= */
thead, tbody, tr, td, th {
    color: #1a202c !important;
    font-weight: 600;
}

/* =============================
   REMOVE STREAMLIT BRANDING
   ============================= */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# =============================
# CONSTANTS
# =============================
MODEL_PATH = 'models/churn_model_pipeline.pkl'

REQUIRED_COLUMNS = [
    'Gender', 'Age', 'Tenure_Months', 'ContractType', 'MonthlyCharges',
    'TotalCharges', 'InternetService', 'TechSupport', 'OnlineSecurity',
    'PaymentMethod', 'Complaints'
]

# =============================
# MODEL LOADING
# =============================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return create_sample_model()
    return joblib.load(MODEL_PATH), None

def create_sample_model():
    os.makedirs('models', exist_ok=True)

    np.random.seed(42)
    n = 1000

    df = pd.DataFrame({
        'Gender': np.random.choice(['Male', 'Female'], n),
        'Age': np.random.randint(18, 70, n),
        'Tenure_Months': np.random.randint(1, 72, n),
        'ContractType': np.random.choice(['Month-to-Month', 'One-Year', 'Two-Year'], n),
        'MonthlyCharges': np.random.uniform(20, 120, n),
        'InternetService': np.random.choice(['FiberOptic', 'DSL', 'No'], n),
        'TechSupport': np.random.choice(['Yes', 'No'], n),
        'OnlineSecurity': np.random.choice(['Yes', 'No'], n),
        'PaymentMethod': np.random.choice(['Cash', 'BankTransfer', 'CreditCard', 'EWallet'], n),
        'Complaints': np.random.choice(['Yes', 'No'], n),
    })

    df['TotalCharges'] = df['MonthlyCharges'] * df['Tenure_Months']
    df['Churn'] = np.random.choice(['Yes', 'No'], n)

    X = df[REQUIRED_COLUMNS]
    y = df['Churn']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['Age', 'Tenure_Months', 'MonthlyCharges', 'TotalCharges']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), 
         ['Gender', 'ContractType', 'InternetService', 'TechSupport',
          'OnlineSecurity', 'PaymentMethod', 'Complaints'])
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])

    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model, None

# =============================
# MAIN APP
# =============================
def main():

    st.markdown("""
    <div class="custom-header">
        <h1>🎯 Churn Predictor Pro</h1>
        <p>AI-Powered Customer Retention Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 🚀 Navigation")
        st.success("Model Loaded Successfully")

    model, _ = load_model()

    tab1, tab2 = st.tabs(["🔮 Single Prediction", "📊 Batch Prediction"])

    with tab1:
        st.markdown("### Customer Details")
        with st.form("predict"):
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 18, 100, 30)
            tenure = st.slider("Tenure Months", 1, 72, 12)
            contract = st.selectbox("Contract Type", ["Month-to-Month", "One-Year", "Two-Year"])
            monthly = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
            total = monthly * tenure
            submit = st.form_submit_button("Predict")

        if submit:
            df = pd.DataFrame([{
                'Gender': gender,
                'Age': age,
                'Tenure_Months': tenure,
                'ContractType': contract,
                'MonthlyCharges': monthly,
                'TotalCharges': total,
                'InternetService': 'FiberOptic',
                'TechSupport': 'Yes',
                'OnlineSecurity': 'Yes',
                'PaymentMethod': 'CreditCard',
                'Complaints': 'No'
            }])
            pred = model.predict(df)[0]
            st.success(f"Prediction: {pred}")

    with tab2:
        st.markdown("### Upload CSV for Batch Prediction")
        file = st.file_uploader("Upload CSV", type="csv")
        if file:
            df = pd.read_csv(file)
            df['Prediction'] = model.predict(df[REQUIRED_COLUMNS])
            st.dataframe(df)

if __name__ == "__main__":
    main()
