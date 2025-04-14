#Classification
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt


# Set the path to the folder containing the CSV
folder_path = "/Users/yifeishang/Desktop/ML_HW2/final"

# Load all .csv files in the folder (or use .csv.gz if compressed)
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Combine all CSVs into one DataFrame
df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

# Show the first 5 rows
print("First 5 records:\n", df.head())

# === Event Type Distribution ===
# event_counts = df['event_type'].value_counts()
# print("\nEvent Type Counts:\n", event_counts)
# event_counts.plot(kind='bar', title='Event Type Distribution', rot=0, color='skyblue')
# plt.xlabel('Event Type')
# plt.ylabel('Count')
# plt.tight_layout()
# plt.show()

# === Brand Type Distribution by event type ===

# # Step 1: Get top N brands by total interactions
# top_n = 10
# top_brands = df['brand'].value_counts().head(top_n).index.tolist()

# # Step 2: Filter data for just those brands
# filtered_df = df[df['brand'].isin(top_brands)]

# # Step 3: Group by event_type and brand, count events
# brand_event = filtered_df.groupby(['brand', 'event_type']).size().unstack(fill_value=0)

# # Step 4: Sort brands by total activity (optional for better readability)
# brand_event = brand_event.loc[brand_event.sum(axis=1).sort_values(ascending=False).index]

# # Step 5: Plot
# brand_event.plot(kind='bar', figsize=(12, 6), width=0.8)

# plt.title('Top 10 Brands by Event Type')
# plt.xlabel('Brand')
# plt.ylabel('Event Count')
# plt.xticks(rotation=45, ha='right')
# plt.legend(title='Event Type')
# plt.tight_layout()
# plt.show()

# === price distribution based on brand ===
top_brands = df['brand'].value_counts().head(10).index.tolist()
filtered = df[df['brand'].isin(top_brands)]
plt.figure(figsize=(12, 6))
filtered.boxplot(column='price', by='brand', grid=False, vert=True)

plt.title('Price Distribution by Brand')
plt.suptitle("")  # removes automatic boxplot title
plt.xlabel('Brand')
plt.ylabel('Price')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()




# Convert to a classification target
# df['event_purchase'] = (df['event_type'] == 'purchase').astype(int)

# features = df.groupby("user_id").agg({
#     'event_type': lambda x: (x == 'view').sum(),
#     'category_code': 'nunique',
#     'brand': 'nunique',
#     'event_purchase': 'max'
# }).rename(columns={
#     'event_type': 'view_count',
#     'category_code': 'unique_categories',
#     'brand': 'unique_brands',
#     'event_purchase': 'label'
# }).reset_index()

# # Train-Test Split and Model

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report

# X = features.drop(columns=['user_id', 'label'])
# y = features['label']

# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3)

# model = RandomForestClassifier()
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# print(classification_report(y_test, y_pred))
