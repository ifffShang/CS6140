# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt

# # Step 1: Load CSV file
# file_path = "/Users/yifeishang/Desktop/ML_HW2/final/2020-Apr.csv"
# df = pd.read_csv(file_path)
# print("CSV Loaded:")
# print(df.head())
# print(df.info())

# # Step 2: Create binary label — 1 if 'purchase', else 0
# df['event_purchase'] = (df['event_type'] == 'purchase').astype(int)

# # Step 3: Aggregate user-level features
# user_df = df.groupby('user_id').agg({
#     'event_type': lambda x: (x == 'view').sum(),        # view_count
#     'category_code': 'nunique',                         # unique_categories
#     'brand': 'nunique',                                 # unique_brands
#     'event_purchase': 'max'                             # label (did they purchase?)
# }).rename(columns={
#     'event_type': 'view_count',
#     'category_code': 'unique_categories',
#     'brand': 'unique_brands',
#     'event_purchase': 'label'
# }).reset_index()

# # Add cart count per user (optional but helpful)
# user_df['cart_count'] = df[df['event_type'] == 'cart'].groupby('user_id').size()
# user_df['cart_count'] = user_df['cart_count'].fillna(0)

# # Step 4: Prepare features and target
# features = ['view_count', 'cart_count', 'unique_categories', 'unique_brands']
# X = user_df[features]
# y = user_df['label']

# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# # Step 5: Train Random Forest classifier
# rf_model = RandomForestClassifier(
#     n_estimators=100,
#     max_depth=10,
#     class_weight='balanced',
#     random_state=42
# )
# rf_model.fit(X_train, y_train)

# # Step 6: Evaluate model
# y_pred = rf_model.predict(X_test)

# print("\nRandom Forest Classification Report:")
# print(classification_report(y_test, y_pred, digits=4))

# # Step 7: Confusion matrix
# ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test, display_labels=["No Purchase", "Purchase"], cmap="Blues")
# plt.title("Confusion Matrix")
# plt.show()

# # Step 8: Feature importance plot
# importances = rf_model.feature_importances_
# plt.barh(features, importances, color='skyblue')
# plt.xlabel("Importance")
# plt.title("Feature Importance (Random Forest)")
# plt.tight_layout()
# plt.show()
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset
file_path = "/Users/yifeishang/Desktop/ML_HW2/final/2020-Apr.csv"
df = pd.read_csv(file_path)
df['event_purchase'] = (df['event_type'] == 'purchase').astype(int)

# Aggregate user-level features
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

# Features and labels
features = ['view_count', 'cart_count', 'unique_categories', 'unique_brands']
X = user_df[features].values
y = user_df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# --- Random Forest From Scratch ---
class RandomForestScratch:
    def __init__(self, n_estimators=10, max_depth=5, max_features='sqrt', random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
        self.feature_indices = []
        self.random_state = random_state
        np.random.seed(self.random_state)

    def _get_feature_subset(self, n_features):
        if self.max_features == 'sqrt':
            return np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)
        elif self.max_features == 'log2':
            return np.random.choice(n_features, int(np.log2(n_features)), replace=False)
        elif isinstance(self.max_features, int):
            return np.random.choice(n_features, self.max_features, replace=False)
        else:
            return np.arange(n_features)  # all features

    def _bootstrap_sample(self, X, y):
        class_0 = np.where(y == 0)[0]
        class_1 = np.where(y == 1)[0]
        
        # Sample equal counts (or ratio-matched)
        n_samples = min(len(class_0), len(class_1))
        indices_0 = np.random.choice(class_0, size=n_samples, replace=True)
        indices_1 = np.random.choice(class_1, size=n_samples, replace=True)
        
        combined = np.concatenate([indices_0, indices_1])
        np.random.shuffle(combined)
        
        return X[combined], y[combined]


    def fit(self, X, y):
        self.trees = []
        self.feature_indices = []
        n_total_features = X.shape[1]

        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            selected_features = self._get_feature_subset(n_total_features)
            tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X_sample[:, selected_features], y_sample)
            self.trees.append(tree)
            self.feature_indices.append(selected_features)

    def predict(self, X):
        # Collect predictions from each tree
        tree_preds = []
        for tree, features in zip(self.trees, self.feature_indices):
            pred = tree.predict(X[:, features])
            tree_preds.append(pred)

        # Majority voting
        tree_preds = np.array(tree_preds)
        y_pred = np.round(np.mean(tree_preds, axis=0)).astype(int)
        return y_pred

# Train and predict
rf = RandomForestScratch(n_estimators=50, max_depth=10, max_features=2)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
