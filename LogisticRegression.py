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
    
class LogisticRegression:
    """Math: Sigmoid = 1/(1+e^-z). Loss = Binary Cross Entropy via Gradient Descent"""
    def __init__(self, lr=0.01, iters=1000):
        self.lr = lr
        self.iters = iters

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        for _ in range(self.iters):
            z = np.dot(X, self.theta)
            h = self._sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

    def predict(self, X):
        return (self._sigmoid(np.dot(X, self.theta)) >= 0.5).astype(int)

model_4 = LogisticRegression()
model_4.fit(X_train, y_c_train)