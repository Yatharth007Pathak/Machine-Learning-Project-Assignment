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

class RegularizedRegression:
    """Ridge Math: beta = (X^T*X + lambda*I)^-1 * X^T*y"""
    def __init__(self, alpha=1.0, method='ridge'):
        self.alpha = alpha
        self.method = method

    def fit(self, X, y):
        X = np.column_stack([np.ones(X.shape[0]), X])
        n_features = X.shape[1]
        if self.method == 'ridge':
            identity = np.eye(n_features)
            identity[0, 0] = 0 # Don't penalize intercept
            self.beta = np.linalg.inv(X.T @ X + self.alpha * identity) @ X.T @ y
        # Lasso implementation usually requires Coordinate Descent math

    def predict(self, X):
        X = np.column_stack([np.ones(X.shape[0]), X])
        return X @ self.beta

model_3 = RegularizedRegression(alpha=0.1)
model_3.fit(X_r_train, y_r_train)