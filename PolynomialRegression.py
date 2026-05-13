import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Telco_customer_churn.csv')

# Engineering Target: CLV = Tenure * Monthly Charge
df['CLV'] = df['Tenure Months'] * df['Monthly Charge']

# Selecting Features (Avoiding leakage for regression)
features = ['Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Paperless Billing', 'Monthly Charge', 'Tenure Months']
X_raw = pd.get_dummies(df[features], drop_first=True).values.astype(float)
y_class = (df['Churn Value'] == 1).values.astype(float)
y_reg = df['CLV'].values.astype(float)

# Manual Feature Scaling: z = (x - mu) / sigma
def manual_scaler(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / (std + 1e-9)

X_scaled = manual_scaler(X_raw)

# Split data (80/20 split)
def split_data(X, y):
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    limit = int(len(idx) * 0.8)
    return X[idx[:limit]], X[idx[limit:]], y[idx[:limit]], y[idx[limit:]]

X_train, X_test, y_c_train, y_c_test = split_data(X_scaled, y_class)
X_r_train, X_r_test, y_r_train, y_r_test = split_data(X_scaled, y_reg)

class PolynomialRegression:
    """Math: Transform X to [1, x, x^2] then solve via OLS"""
    def fit(self, X, y, degree=2):
        X_poly = X.copy()
        for d in range(2, degree + 1):
            X_poly = np.column_stack([X_poly, X**d])
        X_poly = np.column_stack([np.ones(X_poly.shape[0]), X_poly])
        self.beta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
        self.degree = degree

    def predict(self, X):
        X_poly = X.copy()
        for d in range(2, self.degree + 1):
            X_poly = np.column_stack([X_poly, X**d])
        X_poly = np.column_stack([np.ones(X_poly.shape[0]), X_poly])
        return X_poly @ self.beta

model_2 = PolynomialRegression()
model_2.fit(X_r_train, y_r_train)