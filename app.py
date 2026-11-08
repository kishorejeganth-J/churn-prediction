import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Predictor")

@st.cache_resource
def load_artifact():
    return joblib.load("churn_model.pkl")

art = load_artifact()
model, columns, threshold = art["model"], art["columns"], art["threshold"]

st.title("Customer Churn Predictor")
st.write("Enter customer details to estimate the risk of leaving.")

c1, c2 = st.columns(2)
with c1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone service", ["Yes", "No"])
    multiple = st.selectbox("Multiple lines",
                            ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
with c2:
    opts = ["No", "Yes", "No internet service"]
    security = st.selectbox("Online security", opts)
    backup = st.selectbox("Online backup", opts)
    protection = st.selectbox("Device protection", opts)
    support = st.selectbox("Tech support", opts)
    tv = st.selectbox("Streaming TV", opts)
    movies = st.selectbox("Streaming movies", opts)
    contract = st.selectbox("Contract",
                            ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly charges", 18.0, 120.0, 70.0)

total = tenure * monthly
st.caption(f"Total charges estimated as tenure x monthly = {total:.2f}")

if st.button("Predict churn risk"):
    row = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone, "MultipleLines": multiple,
        "InternetService": internet, "OnlineSecurity": security,
        "OnlineBackup": backup, "DeviceProtection": protection,
        "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly,
        "TotalCharges": total}])
    x = pd.get_dummies(row).reindex(columns=columns, fill_value=0)
    x = x.astype(float)
    prob = model.predict_proba(x)[0, 1]
    st.metric("Churn probability", f"{prob:.1%}")
    if prob >= threshold:
        st.error("High risk: this customer is likely to leave.")
    else:
        st.success("Low risk: this customer is likely to stay.")
    st.caption(f"Logistic regression, decision threshold {threshold}")