import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# Step 1: Load the raw clickstream data
file_path = "/Users/yifeishang/Desktop/ML_HW2/final/2020-Apr.csv"
df = pd.read_csv(file_path)

# Step 2: Create 'label' column: 1 if event_type is 'purchase', else 0
df['event_purchase'] = (df['event_type'] == 'purchase').astype(int)

# Step 3: Aggregate features by user_id
user_df = df.groupby('user_id').agg({
    'event_type': lambda x: (x == 'view').sum(),        # view_count
    'category_code': 'nunique',                         # unique_categories
    'brand': 'nunique',                                 # unique_brands
    'event_purchase': 'max'                             # label
}).rename(columns={
    'event_type': 'view_count',
    'category_code': 'unique_categories',
    'brand': 'unique_brands',
    'event_purchase': 'label'
}).reset_index()

# Optional: add cart count if needed
user_df['cart_count'] = df[df['event_type'] == 'cart'].groupby('user_id').size()
user_df['cart_count'] = user_df['cart_count'].fillna(0)

# Step 4: Prepare training data
features = ['view_count', 'cart_count', 'unique_categories', 'unique_brands']
X = user_df[features]
y = user_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# Step 5: Train Logistic Regression using SGD
model = make_pipeline(
    StandardScaler(),
    SGDClassifier(loss='log_loss', penalty='l2', max_iter=1000, tol=1e-3, class_weight='balanced', random_state=42)
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 6: Evaluation
print("Logistic Regression (SGDClassifier) Results:\n")
print(classification_report(y_test, y_pred, digits=4))

