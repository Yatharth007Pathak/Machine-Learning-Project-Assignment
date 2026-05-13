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

class NaiveBayes:
    """Math: P(y|X) = [P(X|y) * P(y)] / P(X) using Gaussian PDF"""
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.stats = []
        for c in self.classes:
            X_c = X[y == c]
            self.stats.append([(np.mean(col), np.var(col)) for col in X_c.T])

    def _pdf(self, x, mean, var):
        exponent = np.exp(-((x - mean)**2) / (2 * var + 1e-9))
        return (1 / np.sqrt(2 * np.pi * var + 1e-9)) * exponent

    def predict(self, X):
        y_pred = []
        for x in X:
            probs = []
            for i, c in enumerate(self.classes):
                prior = np.log(1/len(self.classes))
                likelihood = np.sum(np.log([self._pdf(x[j], m, v) for j, (m, v) in enumerate(self.stats[i])]))
                probs.append(prior + likelihood)
            y_pred.append(self.classes[np.argmax(probs)])
        return np.array(y_pred)

model_5 = NaiveBayes()
model_5.fit(X_train, y_c_train)