# Bank Customer Churn Prediction - Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load Excel Dataset
# -------------------------------
xls = pd.ExcelFile('Churn_Modelling.xlsx')
print("Sheets in Excel file:", xls.sheet_names)

# Load the main sheet (replace index 0 with your sheet if different)
df = pd.read_excel('Churn_Modelling.xlsx', sheet_name='Churn_Modelling', header=0)

# Strip column names
df.columns = df.columns.str.strip()
print("Columns in dataset:", df.columns)

# -------------------------------
# 2. Prepare Features & Target
# -------------------------------
X = df.drop(['Exited', 'RowNumber', 'CustomerId', 'Surname'], axis=1)
y = df['Exited']

# Convert boolean columns to int
bool_cols = ['HasCrCard', 'IsActiveMember', 'IsHighValueCustomer']
for col in bool_cols:
    X[col] = X[col].astype(int)

# One-Hot Encode AgeGroup
categorical_cols = ['AgeGroup']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# -------------------------------
# 3. Split into Train and Test
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 4. Scale Numeric Features
# -------------------------------
num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'BalanceSalaryRatio']
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# -------------------------------
# 5. Train Logistic Regression
# -------------------------------
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)

# -------------------------------
# 6. Make Predictions
# -------------------------------
y_pred = lr.predict(X_test)
y_prob = lr.predict_proba(X_test)[:,1]

# -------------------------------
# 7. Evaluate Model
# -------------------------------
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)

roc_score = roc_auc_score(y_test, y_prob)
print("\nROC-AUC Score:", roc_score)

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label='Logistic Regression (AUC = {:.2f})'.format(roc_score))
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# -------------------------------
# 8. Feature Importance
# -------------------------------
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': lr.coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\n=== Feature Importance (by absolute coefficient) ===")
print(coef_df)
