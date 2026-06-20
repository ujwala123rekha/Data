# =====================================================
# RETAIL SALES ANALYSIS & PREDICTION PROJECT
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("SampleSuperstore.csv")

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATASET INFO =====")
print(df.info())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== SHAPE BEFORE CLEANING =====")
print(df.shape)

# =====================================================
# DATA CLEANING
# =====================================================

df.drop_duplicates(inplace=True)

print("\n===== SHAPE AFTER CLEANING =====")
print(df.shape)

# =====================================================
# EXPLORATORY DATA ANALYSIS
# =====================================================

# Sales by Category
sales_category = df.groupby('Category')['Sales'].sum()

plt.figure(figsize=(8,5))
sales_category.plot(kind='bar')
plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()

# Profit by Region
profit_region = df.groupby('Region')['Profit'].sum()

plt.figure(figsize=(8,5))
profit_region.plot(kind='bar')
plt.title("Profit by Region")
plt.xlabel("Region")
plt.ylabel("Profit")
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))

sns.heatmap(
    df[['Sales', 'Profit', 'Quantity', 'Discount']].corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# =====================================================
# FEATURE ENGINEERING
# =====================================================

columns_to_encode = []

for col in ['Category', 'Region', 'Segment']:
    if col in df.columns:
        columns_to_encode.append(col)

df = pd.get_dummies(
    df,
    columns=columns_to_encode,
    drop_first=True
)

# =====================================================
# PREPARE FEATURES & TARGET
# =====================================================

drop_columns = [
    'Sales',
    'Order ID',
    'Customer Name',
    'Product Name',
    'Order Date',
    'Ship Date',
    'Customer ID',
    'Postal Code',
    'City',
    'State',
    'Country',
    'Product ID'
]

existing_columns = [col for col in drop_columns if col in df.columns]

X = df.drop(columns=existing_columns)
y = df['Sales']

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# =====================================================
# MODEL TRAINING
# =====================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# =====================================================
# PREDICTIONS
# =====================================================

y_pred = model.predict(X_test)

# =====================================================
# EVALUATION
# =====================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")
print("MAE       :", round(mae, 2))
print("MSE       :", round(mse, 2))
print("RMSE      :", round(rmse, 2))
print("R2 Score  :", round(r2, 4))

# =====================================================
# ACTUAL VS PREDICTED
# =====================================================

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.tight_layout()
plt.show()

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

print("\n===== TOP 10 IMPORTANT FEATURES =====")
print(importance_df.head(10))

plt.figure(figsize=(10,6))

plt.barh(
    importance_df['Feature'][:10],
    importance_df['Importance'][:10]
)

plt.title("Top 10 Important Features")
plt.xlabel("Importance")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# =====================================================
# BUSINESS INSIGHTS
# =====================================================

print("\n===== BUSINESS INSIGHTS =====")
print("1. Identify the category generating maximum sales.")
print("2. Analyze regions contributing highest profit.")
print("3. Observe impact of discount on profit.")
print("4. Understand key features affecting sales.")
print("5. Use predictions for future sales planning.")

print("\n===== PROJECT COMPLETED SUCCESSFULLY =====")
