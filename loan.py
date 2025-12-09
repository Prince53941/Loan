def prediction_form(df, meta, model):
    st.subheader("Predict Loan Approval for a New Applicant")

    with st.form("prediction_form"):
        user_input = {}

        for col in meta["original_columns"]:

            if col == meta["target_col"]:
                continue

            if col in meta["numeric_cols"]:

                user_input[col] = st.number_input(
                    f"{col} (numeric)",
                    value=float(df[col].median()) if df[col].median() is not None else 0.0
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

            if "Approved" in nice_text:
                st.success("Yes, the loan will be approved.")
            elif "Not Approved" in nice_text:
                st.error("No, the loan will not be approved.")
            else:
                st.info(f"Prediction: {nice_text}")

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_single)[0]
                classes = list(model.classes_)

                if "Yes" in classes:
                    idx = classes.index("Yes")
                    approval_prob = proba[idx]
                else:
                    approval_prob = max(proba)

                st.write(f"Model confidence: {approval_prob * 100:.2f}%")

        except Exception as e:
            st.error(f"Error in prediction: {e}")
