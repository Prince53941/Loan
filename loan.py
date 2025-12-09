import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# =========================================
# Streamlit Page Config
# =========================================
st.set_page_config(
    page_title="Loan Approval Automation System",
    page_icon="💳",
    layout="wide"
)

# =========================================
# Helper Functions
# =========================================
def basic_eda(df, target_col):
    st.subheader("Dataset Overview")
    st.write("Shape of data:", df.shape)
    st.write("First 5 rows:")
    st.dataframe(df.head())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    if target_col in df.columns:
        st.subheader(f"Target Variable Distribution: {target_col}")
        st.write(df[target_col].value_counts())
        st.bar_chart(df[target_col].value_counts())


def preprocess_data(df, target_col):
    # Drop rows where target is missing
    df = df.dropna(subset=[target_col])

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Handle missing values
    # Numeric: fill with median
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())

    # Categorical: fill with mode
    for col in categorical_cols:
        if X[col].isnull().sum() > 0:
            X[col] = X[col].fillna(X[col].mode()[0])

    # One-hot encode categorical variables
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale numeric features (helpful for Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # Numeric columns keep the same names after encoding
    numeric_encoded_cols = [col for col in numeric_cols if col in X_train.columns]

    if numeric_encoded_cols:
        X_train_scaled[numeric_encoded_cols] = scaler.fit_transform(X_train[numeric_encoded_cols])
        X_test_scaled[numeric_encoded_cols] = scaler.transform(X_test[numeric_encoded_cols])

    meta = {
        "original_columns": X.columns.tolist(),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_columns": X_encoded.columns.tolist(),
        "scaler": scaler,
        "numeric_encoded_cols": numeric_encoded_cols,
        "target_col": target_col,
    }

    return X_train_scaled, X_test_scaled, y_train, y_test, meta


def train_model(algorithm, X_train, y_train):
    if algorithm == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif algorithm == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.subheader("Model Evaluation")
    st.write(f"**Accuracy:** {acc:.4f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.text("Confusion Matrix:")
    st.write(pd.DataFrame(cm))


def prepare_single_input(meta, input_dict):
    """
    Convert user input dict to a single-row DataFrame with the same
    columns as training data (after get_dummies).
    """
    df_single = pd.DataFrame([input_dict])

    # Handle missing like before
    for col in meta["numeric_cols"]:
        if col in df_single.columns:
            df_single[col] = df_single[col].fillna(df_single[col].median())

    for col in meta["categorical_cols"]:
        if col in df_single.columns:
            if df_single[col].isnull().sum() > 0:
                df_single[col] = df_single[col].fillna(df_single[col].mode()[0])

    # One-hot encode
    df_single_encoded = pd.get_dummies(
        df_single,
        columns=meta["categorical_cols"],
        drop_first=True
    )

    # Align with training columns
    df_single_encoded = df_single_encoded.reindex(
        columns=meta["feature_columns"],
        fill_value=0
    )

    # Scale numeric columns
    if meta["numeric_encoded_cols"]:
        df_single_encoded[meta["numeric_encoded_cols"]] = meta["scaler"].transform(
            df_single_encoded[meta["numeric_encoded_cols"]]
        )

    return df_single_encoded


def pretty_loan_output(raw_pred):
    """
    Map raw model output to a user-friendly message.
    Adjust mapping based on your dataset labels.
    """
    # Common cases: 'Y'/'N', 1/0, 'Approved'/'Rejected'
    if str(raw_pred) in ["Y", "1", "Approved", "Yes"]:
        return "Loan Approved"
    elif str(raw_pred) in ["N", "0", "Rejected", "No"]:
        return "Loan Not Approved"
    else:
        # Fallback – just show original value
        return str(raw_pred)


def prediction_form(df, meta, model):
    st.subheader("Predict Loan Approval for a New Applicant")

    with st.form("prediction_form"):
        user_input = {}

        for col in meta["original_columns"]:
            # Do NOT ask for target column (Loan_Status)
            if col == meta["target_col"]:
                continue

            if col in meta["numeric_cols"]:
                col_min = float(df[col].min())
                col_max = float(df[col].max())
                default_val = float(df[col].median())

                if col_min == col_max:
                    col_min = col_min * 0.5
                    col_max = col_max * 1.5

                user_input[col] = st.number_input(
                    f"{col} (numeric)",
                    min_value=col_min,
                    max_value=col_max,
                    value=default_val
                )

            elif col in meta["categorical_cols"]:
                options = df[col].dropna().unique().tolist()
                if len(options) == 0:
                    options = ["N/A"]
                user_input[col] = st.selectbox(
                    f"{col} (category)",
                    options=options,
                    index=0
                )

            else:
                user_input[col] = st.text_input(f"{col} (text)")

        submitted = st.form_submit_button("Predict Loan Approval")

    if submitted:
        try:
            X_single = prepare_single_input(meta, user_input)
            prediction = model.predict(X_single)[0]

            nice_text = pretty_loan_output(prediction)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_single)[0]
                st.success(f"Prediction: {nice_text}  (raw label: {prediction})")
                st.write("Prediction Probabilities:")
                st.write({str(cls): float(p) for cls, p in zip(model.classes_, proba)})
            else:
                st.success(f"Prediction: {nice_text}  (raw label: {prediction})")

        except Exception as e:
            st.error(f"Error in prediction: {e}")


# =========================================
# Main App
# =========================================
def main():
    st.title("💳 Loan Approval Automation System")
    st.markdown(
        """
        This app builds a **Loan Approval Prediction Model** using your dataset.

        **Steps:**
        1. Upload your loan dataset (CSV).
        2. App will use `Loan_Status` as the target (if present).
        3. Choose ML algorithm and train the model.
        4. View evaluation metrics.
        5. Predict loan approval for a new applicant.
        """
    )

    st.sidebar.header("Upload & Settings")

    uploaded_file = st.sidebar.file_uploader("Upload your loan CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Force/Prefer Loan_Status as target if present
        if "Loan_Status" in df.columns:
            target_col = "Loan_Status"
        else:
            # Fallback: let user choose
            st.sidebar.warning("Column `Loan_Status` not found. Please select the target column manually.")
            target_col = st.sidebar.selectbox("Select Target Column", options=df.columns)

        algorithm = st.sidebar.selectbox("Select Algorithm", ["Logistic Regression", "Random Forest"])

        st.sidebar.markdown("---")
        train_button = st.sidebar.button("Run Full Pipeline (Clean + Train + Evaluate)")

        # Show EDA
        st.header("1. Exploratory Data Analysis")
        basic_eda(df, target_col)

        if train_button:
            with st.spinner("Preprocessing data and training model..."):
                X_train, X_test, y_train, y_test, meta = preprocess_data(df.copy(), target_col)
                model = train_model(algorithm, X_train, y_train)

                # Store in session_state for later prediction
                st.session_state["df"] = df
                st.session_state["meta"] = meta
                st.session_state["model"] = model

            st.success("Model trained successfully!")

            st.header("2. Model Performance")
            evaluate_model(model, X_test, y_test)

        # If model is already trained, show prediction UI
        if "model" in st.session_state and "meta" in st.session_state and "df" in st.session_state:
            st.header("3. Loan Approval Prediction")
            prediction_form(st.session_state["df"], st.session_state["meta"], st.session_state["model"])

    else:
        st.info("Please upload a CSV file from the sidebar to start.")


if __name__ == "__main__":
    main()
