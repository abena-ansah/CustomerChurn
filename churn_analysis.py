import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the data
df = pd.read_csv("TelcoCustomerChurnData.csv")

# show info block
print("\n=== Data Info ===")
print(df.info())

# show data preview
print("\n=== Data Head ===")
print(df.head())

# Create tenure groups
df['TenureGroup'] = pd.cut(
    df['tenure'],
    bins=[0, 12, 24, 48, 72],
    labels=['0-12m', '12-24m', '24-48m', '48-72m']
)

# Show churn proportions
print("\n=== Churn Proportions By Group ===")
print(df.groupby('TenureGroup')['Churn'].value_counts(normalize=True))
print(df.groupby('Contract')['Churn'].value_counts(normalize=True))
print(df.groupby('PaymentMethod')['Churn'].value_counts(normalize=True))
print(df.groupby('OnlineSecurity')['Churn'].value_counts(normalize=True))
print(df.groupby('TechSupport')['Churn'].value_counts(normalize=True))

# Plot churn distribution
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.savefig('churn_distribution.png')
plt.close()

# Plot churn by contract
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title('Churn by Contract Type')
plt.savefig('churn_by_contract.png')
plt.close()

# Plot churn by tenure
sns.countplot(x='TenureGroup', hue='Churn', data=df)
plt.title('Churn by Tenure Group')
plt.savefig('churn_by_tenure_group.png')
plt.close()

# Plot churn by payment method
plt.figure(figsize=(10,5))
sns.countplot(y='PaymentMethod', hue='Churn', data=df)
plt.title('Churn by Payment Method')
plt.savefig('churn_by_payment_method.png')
plt.close()

# Plot churn by OnlineSecurity
sns.countplot(x='OnlineSecurity', hue='Churn', data=df)
plt.title('Churn by Online Security')
plt.savefig('churn_by_onlinesecurity.png')
plt.close()

# Prepare data for modeling
df_model = df.copy()
df_model.drop(['customerID', 'TotalCharges'], axis=1, inplace=True)
df_model = pd.get_dummies(df_model, drop_first=True)

X = df_model.drop('Churn_Yes', axis=1)
y = df_model['Churn_Yes']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\n=== Logistic Regression Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred_lr))

print("\n=== Logistic Regression Classification Report ===")
print(classification_report(y_test, y_pred_lr))

# Print top logistic regression coefficients
coefs = pd.Series(lr.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
print("\nTop Logistic Regression Coefficients:")
print(coefs.head(10))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n=== Random Forest Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred_rf))

print("\n=== Random Forest Classification Report ===")
print(classification_report(y_test, y_pred_rf))

# Random forest feature importances
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop Random Forest Features:")
print(importances.head(10))

# Save feature importance plot
plt.figure(figsize=(8,5))
sns.barplot(x=importances.head(10), y=importances.head(10).index)
plt.title("Random Forest Top Features")
plt.savefig("rf_feature_importances.png")
plt.close()


