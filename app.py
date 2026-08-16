import streamlit as st
import pandas as pd
try:
    import joblib
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib==1.5.1"])
    import joblib

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")


# =========================================================
# TITLE
# =========================================================

st.title("🏦 Bank Customer Churn Prediction")

st.markdown(
    """
    ### Predictive Modeling and Risk Scoring for Bank Customer Churn

    This application uses a **Gradient Boosting model** to estimate
    the probability that a bank customer will churn.
    """
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("👤 Customer Information")

credit_score = st.sidebar.slider(
    "Credit Score",
    300,
    900,
    650
)

age = st.sidebar.slider(
    "Age",
    18,
    100,
    40
)

tenure = st.sidebar.slider(
    "Tenure (Years)",
    0,
    10,
    5
)

balance = st.sidebar.number_input(
    "Account Balance",
    min_value=0.0,
    value=50000.0,
    step=1000.0
)

num_products = st.sidebar.slider(
    "Number of Products",
    1,
    4,
    1
)

has_cr_card = st.sidebar.selectbox(
    "Has Credit Card?",
    ["Yes", "No"]
)

active_member = st.sidebar.selectbox(
    "Is Active Member?",
    ["Yes", "No"]
)

estimated_salary = st.sidebar.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=75000.0,
    step=1000.0
)

geography = st.sidebar.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"]
)


# =========================================================
# CONVERT CATEGORICAL VALUES
# =========================================================

has_cr_card_value = 1 if has_cr_card == "Yes" else 0

active_member_value = 1 if active_member == "Yes" else 0


# =========================================================
# FEATURE ENGINEERING
# =========================================================

balance_salary_ratio = balance / (estimated_salary + 1)

product_density = num_products / (tenure + 1)

age_tenure = age * tenure

engagement_score = (
    has_cr_card_value
    + active_member_value
    + num_products
)


# =========================================================
# CREATE CUSTOMER DATA
# =========================================================

customer = pd.DataFrame({
    "CreditScore": [credit_score],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_products],
    "HasCrCard": [has_cr_card_value],
    "IsActiveMember": [active_member_value],
    "EstimatedSalary": [estimated_salary],

    "BalanceSalaryRatio": [balance_salary_ratio],
    "ProductDensity": [product_density],
    "AgeTenure": [age_tenure],
    "EngagementScore": [engagement_score],

    "Geography_Germany": [
        1 if geography == "Germany" else 0
    ],

    "Geography_Spain": [
        1 if geography == "Spain" else 0
    ],

    "Gender_Male": [
        1 if gender == "Male" else 0
    ]
})


# =========================================================
# ENSURE EXACT MODEL COLUMN ORDER
# =========================================================

customer = customer[model_columns]


# =========================================================
# SHOW INPUT DATA
# =========================================================

st.subheader("📋 Customer Information")

display_data = customer.copy()

st.dataframe(
    display_data,
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

st.divider()

st.subheader("🔮 Churn Prediction")


if st.button(
    "Predict Churn Risk",
    type="primary",
    use_container_width=True
):

    # Prediction
    prediction = model.predict(customer)[0]

    # Churn probability
    probability = model.predict_proba(customer)[0][1]

    probability_percent = probability * 100


    # =====================================================
    # RISK LEVEL
    # =====================================================

    if probability < 0.30:

        risk_level = "Low Risk"
        risk_color = "🟢"
        message = "Customer has a low probability of churn."

    elif probability < 0.70:

        risk_level = "Medium Risk"
        risk_color = "🟡"
        message = "Customer has a moderate probability of churn."

    else:

        risk_level = "High Risk"
        risk_color = "🔴"
        message = "Customer has a high probability of churn. Retention action is recommended."


    # =====================================================
    # RESULTS
    # =====================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Churn Probability",
            f"{probability_percent:.2f}%"
        )


    with col2:

        if prediction == 1:

            st.metric(
                "Prediction",
                "Likely to Churn"
            )

        else:

            st.metric(
                "Prediction",
                "Likely to Stay"
            )


    with col3:

        st.metric(
            "Risk Level",
            f"{risk_color} {risk_level}"
        )


    st.divider()


    # =====================================================
    # RISK MESSAGE
    # =====================================================

    if risk_level == "Low Risk":

        st.success(
            f"🟢 **LOW RISK** — {message}"
        )

    elif risk_level == "Medium Risk":

        st.warning(
            f"🟡 **MEDIUM RISK** — {message}"
        )

    else:

        st.error(
            f"🔴 **HIGH RISK** — {message}"
        )


    # =====================================================
    # PROBABILITY BAR
    # =====================================================

    st.subheader("📊 Churn Probability")

    st.progress(
        float(probability)
    )

    st.write(
        f"Estimated churn probability: "
        f"**{probability_percent:.2f}%**"
    )


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.divider()

st.subheader("📈 Model Feature Importance")

feature_importance = pd.DataFrame({

    "Feature": model_columns,

    "Importance": model.feature_importances_

})


feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


st.bar_chart(
    feature_importance.set_index("Feature")
)


st.subheader("Top 10 Churn Drivers")

st.dataframe(
    feature_importance.head(10),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Predictive Modeling and Risk Scoring for Bank Customer Churn | "
    "Gradient Boosting Model"
)
