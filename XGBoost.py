import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
)
# --------------------------
# Step 1: Load and preprocess data
# --------------------------
file_path = "/Users/yifeishang/Desktop/ML_HW2/final/2020-Apr.csv"
df = pd.read_csv(file_path)
df['event_purchase'] = (df['event_type'] == 'purchase').astype(int)

user_df = df.groupby('user_id').agg({
    'event_type': lambda x: (x == 'view').sum(),
    'category_code': 'nunique',
    'brand': 'nunique',
    'event_purchase': 'max'
}).rename(columns={
    'event_type': 'view_count',
    'category_code': 'unique_categories',
    'brand': 'unique_brands',
    'event_purchase': 'label'
}).reset_index()

user_df['cart_count'] = df[df['event_type'] == 'cart'].groupby('user_id').size()
user_df['cart_count'] = user_df['cart_count'].fillna(0)

features = ['view_count', 'cart_count', 'unique_categories', 'unique_brands']
X = user_df[features].values
y = user_df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# --------------------------
# Step 2: Define XGBoost-like classifier with softened class weighting
# --------------------------
class SimpleXGBoost:
    def __init__(self, n_estimators=30, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        # Initialize with log-odds (prior probability)
        pos_ratio = np.clip(y.mean(), 1e-5, 1 - 1e-5)
        y_pred = np.full(len(y), np.log(pos_ratio / (1 - pos_ratio)))
        self.trees = []

        # Calculate and soften scale_pos_weight
        scale_pos_weight = (y == 0).sum() / (y == 1).sum()
        softened_weight = min(scale_pos_weight * 0.5, 2.0)
        print(f"Using softened scale_pos_weight = {softened_weight:.2f}")

        for _ in range(self.n_estimators):
            p = self.sigmoid(y_pred)
            weights = np.where(y == 1, softened_weight, 1.0)
            gradient = weights * (p - y)

            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, -gradient)
            update = tree.predict(X)

            y_pred += self.learning_rate * update
            self.trees.append(tree)

    def predict_proba(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return self.sigmoid(y_pred)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

# --------------------------
# Step 3: Train and evaluate
# --------------------------
model = SimpleXGBoost(n_estimators=30, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
# --------------------------
# Step 4: Output results
# --------------------------
print("Predicted class distribution:", dict(zip(*np.unique(y_pred, return_counts=True))))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 5: Plots
# --------------------------

# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["No Purchase", "Purchase"], cmap="Blues")
plt.title("Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.tight_layout()
plt.show()