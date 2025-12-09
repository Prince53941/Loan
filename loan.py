import pandas as pd
import numpy as np

# 1. Setup Random Seed for Reproducibility
np.random.seed(42)
n = 3000

# 2. Generate Random Basic Features
Gender = np.random.choice(["Male", "Female"], n)
Married = np.random.choice(["Yes", "No"], n)
Dependents = np.random.choice(["0", "1", "2", "3+"], n)
Education = np.random.choice(["Graduate", "Not Graduate"], n, p=[0.7, 0.3])
Self_Employed = np.random.choice(["Yes", "No"], n, p=[0.1, 0.9])

ApplicantIncome = np.random.randint(2000, 30000, n)
CoapplicantIncome = np.random.randint(0, 20000, n)
LoanAmount = np.random.randint(50, 600, n)          # in thousands
Loan_Amount_Term = np.random.choice(
    [180, 240, 300, 360, 480], n, p=[0.1, 0.15, 0.25, 0.4, 0.1]
)
# Note: Credit_History 1.0 = Good, 0.0 = Bad
Credit_History = np.random.choice([1.0, 0.0], n, p=[0.8, 0.2])
Property_Area = np.random.choice(["Urban", "Semiurban", "Rural"], n, p=[0.5, 0.3, 0.2])

# 3. Create Initial DataFrame
df = pd.DataFrame({
    "Gender": Gender,
    "Married": Married,
    "Dependents": Dependents,
    "Education": Education,
    "Self_Employed": Self_Employed,
    "ApplicantIncome": ApplicantIncome,
    "CoapplicantIncome": CoapplicantIncome,
    "LoanAmount": LoanAmount,
    "Loan_Amount_Term": Loan_Amount_Term,
    "Credit_History": Credit_History,
    "Property_Area": Property_Area,
})

# --------------------------
# 4. RULE-BASED LOAN STATUS LOGIC
# --------------------------
# Lower score = safer = Yes
# Higher score = risky = No

# Calculate Total Income
income_total = df['ApplicantIncome'] + 0.5 * df['CoapplicantIncome']

# Calculate Debt-to-Income Ratio (Loan / Monthly Income)
# We multiply LoanAmount by 1000 because it is in 'thousands'
debt_to_income = (df['LoanAmount'] * 1000) / (income_total + 1)  # +1 to avoid division by zero error

risk_score = np.zeros(n)

# Rule 1: Bad credit history increases risk significantly (+3 points)
risk_score += np.where(df['Credit_History'] == 0.0, 3.0, -1.0)

# Rule 2: High debt-to-income is risky
# If Ratio > 25 (Very High Risk) -> +2.0
# If Ratio > 15 (Moderate Risk) -> +1.0
# Else -> -1.0 (Safe)
risk_score += np.where(debt_to_income > 25, 2.0, 
              np.where(debt_to_income > 15, 1.0, -1.0))

# Rule 3: Very low income is risky
risk_score += np.where(df['ApplicantIncome'] < 5000, 1.5, 0.0)

# Rule 4: Education & Area (Minor factors)
risk_score += np.where(df['Education'] == "Not Graduate", 0.5, -0.3)
risk_score += np.where(df['Property_Area'] == "Rural", 0.5, 0.0)

# 5. Assign Final Label
# If risk_score > 1.5, we Reject (No), otherwise Approve (Yes)
df["Loan_Status"] = np.where(risk_score > 1.5, "No", "Yes")

# 6. Check Distribution and Save
print("Dataset Distribution:")
print(df["Loan_Status"].value_counts())

df.to_csv("loan_rule_based_yes_no.csv", index=False)
print("Success! File saved as 'loan_rule_based_yes_no.csv'")
