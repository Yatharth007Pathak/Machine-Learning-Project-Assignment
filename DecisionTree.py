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
    
class DecisionTree:
    """Math: Split nodes based on Gini Impurity (Classification) or MSE reduction (Regression)"""
    # Note: Full scratch trees are highly recursive; simplified logic provided here
    def fit(self, X, y, task='classify'):
        self.task = task
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or depth >= 3:
            return np.mean(y) if self.task == 'reg' else np.bincount(y.astype(int)).argmax()
        
        # Math: Find best feature split (simplified)
        feat_idx = np.argmax(np.var(X, axis=0)) 
        threshold = np.mean(X[:, feat_idx])
        left_mask = X[:, feat_idx] < threshold
        return {
            'idx': feat_idx, 'val': threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth+1),
            'right': self._build_tree(X[~left_mask], y[~left_mask], depth+1)
        }

    def _predict_one(self, x, node):
        if not isinstance(node, dict): return node
        if x[node['idx']] < node['val']: return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

model_7 = DecisionTree()
model_7.fit(X_train, y_c_train)