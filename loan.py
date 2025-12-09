import pandas as pd
import numpy as np

np.random.seed(42)
n = 3000

# Basic features
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
Credit_History = np.random.choice([1.0, 0.0], n, p=[0.8, 0.2])
Property_Area = np.random.choice(["Urban", "Semiurban", "Rural"], n, p=[0.5, 0.3, 0.2])

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
# RULE-BASED LOAN STATUS
# --------------------------
# Lower score = safer = Yes
# Higher score = risky = No

income_total = ApplicantIncome + 0.5 * CoapplicantIncome
debt_to_income = LoanAmount * 1000 / (income_total + 1)  # avoid /0

risk_score = 0.0

# Bad credit history increases risk a lot
risk_score += np.where(Credit_History == 0.0, 3.0, -1.0)

# High debt-to-income is risky
risk_score += np.where(debt_to_income > 25, 2.0,
              np.where(debt_to_income > 15, 1.0, -1.0))

# Very low income is risky
risk_score += np.where(ApplicantIncome < 5000, 1.5, 0.0)

# Education & area: small effects
risk_score += np.where(Education == "Not Graduate", 0.5, -0.3)
risk_score += np.where(Property_Area == "Rural", 0.5, 0.0)

# Final label: if risk_score > 1.5 → No, else Yes
Loan_Status = np.where(risk_score > 1.5, "No", "Yes")
df["Loan_Status"] = Loan_Status

print(df["Loan_Status"].value_counts())

df.to_csv("loan_rule_based_yes_no.csv", index=False)
print("Saved as loan_rule_based_yes_no.csv")
