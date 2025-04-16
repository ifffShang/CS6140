import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --------------------------
# Step 1: Load and preprocess the dataset
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

# Prepare data
features = ['view_count', 'cart_count', 'unique_categories', 'unique_brands']
X = user_df[features].values
y = user_df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# --------------------------
# Step 2: Naïve Bayes from scratch
# --------------------------
class NaiveBayesScratch:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0) + 1e-6  # Avoid divide-by-zero
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def _gaussian_likelihood(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        exponent = -((x - mean) ** 2) / (2 * var)
        return np.exp(exponent) / np.sqrt(2 * np.pi * var)

    def _log_posterior(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.priors[c])
            conditional = np.sum(np.log(self._gaussian_likelihood(c, x)))
            posteriors.append(prior + conditional)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self._log_posterior(x) for x in X])

# --------------------------
# Step 3: Train and evaluate
# --------------------------
model = NaiveBayesScratch()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --------------------------
# Step 4: Output
# --------------------------
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
print("Accuracy:", accuracy_score(y_test, y_pred))
