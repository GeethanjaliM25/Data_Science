import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "churn.csv"


@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def preprocess_data(df):
    df = df.copy()

    df["region_category"] = df["region_category"].fillna(df["region_category"].mode()[0])
    df["preferred_offer_types"] = df["preferred_offer_types"].fillna(df["preferred_offer_types"].mode()[0])
    df["points_in_wallet"] = df["points_in_wallet"].fillna(df["points_in_wallet"].median())
    df.drop_duplicates(inplace=True)

    df["joining_date"] = pd.to_datetime(df["joining_date"])
    df["joining_year"] = df["joining_date"].dt.year
    df["joining_month"] = df["joining_date"].dt.month
    df["joining_day"] = df["joining_date"].dt.day
    df.drop(columns=["joining_date"], inplace=True)

    encoders = {}
    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = ~(((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1))
    df = df[outlier_mask]

    X = df.drop(columns=["churn_risk_score"])
    y = df["churn_risk_score"]
    return X, y, encoders


@st.cache_resource(show_spinner=False)
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy


def transform_user_input(user_inputs, feature_names, encoders):
    row = pd.DataFrame([user_inputs])
    row = row.reindex(columns=feature_names)

    for col in feature_names:
        if col in encoders:
            try:
                row[col] = encoders[col].transform(row[col].astype(str))
            except ValueError:
                row[col] = -1
        else:
            row[col] = pd.to_numeric(row[col], errors="coerce")

    return row.astype(float)


def main():
    st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #fff5f8 0%, #ffe4ee 45%, #ffffff 100%);
            color: #000000;
        }
        .block-container { padding-top: 1.5rem; }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffe8f0 0%, #fff8fb 100%);
            color: #000000;
        }
        .stForm {
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid #f3d0de;
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.06);
        }
        .stButton > button {
            background: linear-gradient(90deg, #f472b6, #fb7185);
            color: white;
            border: none;
            border-radius: 10px;
        }
        .stTextInput > div > div > input,
        .stSelectbox > div > div > div,
        .stNumberInput > div > div > input {
            background-color: white;
            color: black;
        }
        .stMarkdown, .stTextInput label, .stSelectbox label, .stNumberInput label, .stHeader, .stSubheader, .stCaption {
            color: #000000 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🤖📊 Customer Churn Risk Predictor")
    st.caption("Enter customer details in the sidebar and get an instant churn-risk prediction.")

    raw_df = load_data()
    X, y, encoders = preprocess_data(raw_df)
    model, accuracy = train_model(X, y)

    with st.sidebar:
        st.header("🧾 Customer Details")
        st.success(f"Model accuracy: {accuracy:.2%}")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", ["Male", "Female", "Unknown"])
        region_category = st.selectbox("Region Category", ["Village", "City", "Town"])
        membership_category = st.selectbox("Membership Category", ["Basic Membership", "Silver Membership", "Gold Membership", "Platinum Membership", "Premium Membership"])
        joined_through_referral = st.selectbox("Joined Through Referral", ["Yes", "No"])
        preferred_offer_types = st.selectbox("Preferred Offer Types", ["Without Offers", "Credit Card", "Gift Vouchers/Coupons", "Cashback", "Discount"])
        medium_of_operation = st.selectbox("Medium of Operation", ["Desktop", "Mobile", "Both"])
        internet_option = st.selectbox("Internet Option", ["Wi-Fi", "Mobile_Data", "Fiber_Optic"])
        days_since_last_login = st.number_input("Days Since Last Login", min_value=0, max_value=365, value=10)
        avg_time_spent = st.number_input("Average Time Spent", min_value=0.0, max_value=1000.0, value=250.0)
        avg_transaction_value = st.number_input("Average Transaction Value", min_value=0.0, max_value=500000.0, value=1200.0)
        avg_frequency_login_days = st.number_input("Average Frequency Login Days", min_value=0.0, max_value=365.0, value=7.0)
        points_in_wallet = st.number_input("Points in Wallet", min_value=0.0, max_value=100000.0, value=500.0)
        used_special_discount = st.selectbox("Used Special Discount", ["Yes", "No"])
        offer_application_preference = st.selectbox("Offer Application Preference", ["Yes", "No"])
        past_complaint = st.selectbox("Past Complaint", ["Yes", "No"])
        complaint_status = st.selectbox("Complaint Status", ["Not Applicable", "Solved", "Unsolved", "Solved in Follow-up"])
        feedback = st.selectbox("Feedback", ["Poor Website", "Products always in Stock", "Quality Customer Care", "Poor Product Quality", "Too many ads", "User Friendly Website"])
        submitted = st.button("🔮 Predict Churn Risk", use_container_width=True)

    if submitted:
        user_inputs = {
            "age": age,
            "gender": gender,
            "security_no": "SEC-0001",
            "region_category": region_category,
            "membership_category": membership_category,
            "joined_through_referral": joined_through_referral,
            "referral_id": "X01",
            "preferred_offer_types": preferred_offer_types,
            "medium_of_operation": medium_of_operation,
            "internet_option": internet_option,
            "last_visit_time": 1200,
            "days_since_last_login": days_since_last_login,
            "avg_time_spent": avg_time_spent,
            "avg_transaction_value": avg_transaction_value,
            "avg_frequency_login_days": avg_frequency_login_days,
            "points_in_wallet": points_in_wallet,
            "used_special_discount": used_special_discount,
            "offer_application_preference": offer_application_preference,
            "past_complaint": past_complaint,
            "complaint_status": complaint_status,
            "feedback": feedback,
            "joining_year": 2020,
            "joining_month": 6,
            "joining_day": 15,
        }

        prediction_row = transform_user_input(user_inputs, X.columns.tolist(), encoders)
        prediction = model.predict(prediction_row)[0]
        probability = model.predict_proba(prediction_row)[0][1]

        st.subheader("Prediction result")
        if prediction == 1:
            st.markdown(f"<div style='background:#fee2e2;padding:16px;border-radius:12px;color:#991b1b;font-weight:bold;'>⚠️ High churn risk ({probability:.2%})</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:#dcfce7;padding:16px;border-radius:12px;color:#166534;font-weight:bold;'>✅ Low churn risk ({probability:.2%})</div>", unsafe_allow_html=True)
    else:
        st.info("Use the sidebar to enter customer details and predict churn risk.")


if __name__ == "__main__":
    main()
