import streamlit as st
import pandas as pd
import joblib

# PAGE CONFIG
st.set_page_config(
    page_title="AI Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# LOAD MODEL
model = joblib.load("best_churn_model.pkl")

# CUSTOM CSS
st.markdown("""
<style>

/* Main App */
.main {
    padding-top: 1rem;
}

/* Title */
.title {
    font-size: 42px;
    font-weight: 700;
    color: #4F8BF9;
}

/* Subtitle */
.subtitle {
    font-size: 18px;
    color: #BBBBBB;
    margin-bottom: 20px;
}

/* Cards */
.card {
    background-color: #111827;
    padding: 25px;
    border-radius: 16px;
    border: 1px solid #2D3748;
    box-shadow: 0px 0px 12px rgba(0,0,0,0.2);
}

/* Button */
.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    font-size: 20px;
    font-weight: bold;
}

/* Metric cards */
.metric-card {
    background-color: #1F2937;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #374151;
}

.metric-title {
    font-size: 16px;
    color: #9CA3AF;
}

.metric-value {
    font-size: 26px;
    font-weight: bold;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown(
    '<div class="title">📊 AI Customer Churn Prediction Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict telecom customer churn using Machine Learning.</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# SIDEBAR
st.sidebar.title("⚙️ Quick Information")

st.sidebar.info("""
This AI model predicts whether a telecom customer is likely to leave the company.

### Features Used
- Customer demographics
- Internet services
- Billing details
- Subscription details
- Customer support data
""")

st.sidebar.success("Model Loaded Successfully ✅")

# LAYOUT
left_col, right_col = st.columns([1, 1.1])

# LEFT COLUMN → INPUTS
with left_col:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📥 Customer Information")

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.slider(
        "Tenure (Months)",
        0,
        72,
        12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.slider(
        "Monthly Charges",
        0.0,
        200.0,
        50.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0
    )

    st.markdown("<br>", unsafe_allow_html=True)

    predict_button = st.button("🚀 Predict Churn")

    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT COLUMN → RESULTS
with right_col:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📋 Input Data Preview")

    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [senior_citizen],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })

    st.dataframe(
        input_data,
        use_container_width=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if predict_button:

        try:

            prediction = model.predict(input_data)[0]

            probability = model.predict_proba(input_data)[0][1]

            # METRICS

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Churn Probability</div>
                    <div class="metric-value">{probability:.2%}</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:

                risk = "High" if probability > 0.5 else "Low"

                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Risk Level</div>
                    <div class="metric-value">{risk}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # PREDICTION RESULT

            if prediction == 1:

                st.error(
                    "⚠️ Customer is likely to churn."
                )

            else:

                st.success(
                    "✅ Customer is not likely to churn."
                )

            # PROGRESS BAR

            st.subheader("📈 Churn Probability")

            st.progress(float(probability))

            st.caption(
                "Higher probability means greater chance of customer churn."
            )

        except Exception as e:

            st.error(f"Error: {e}")

    else:

        st.info("Fill customer details and click 'Predict Churn'.")

    st.markdown('</div>', unsafe_allow_html=True)

# FOOTER
st.markdown("---")

st.caption(
    "Built with Streamlit • Scikit-learn • Machine Learning"
)
