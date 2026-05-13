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
    
class StackingScratch:
    """Math: Meta-model trained on base-model output feature space"""
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model

    def fit(self, X, y):
        # Create new feature matrix from base model predictions
        meta_X = np.column_stack([m.predict(X) for m in self.base_models])
        self.meta_model.fit(meta_X, y)

    def predict(self, X):
        meta_X = np.column_stack([m.predict(X) for m in self.base_models])
        return self.meta_model.predict(meta_X)

model_14 = StackingScratch([model_4, model_5], model_4)