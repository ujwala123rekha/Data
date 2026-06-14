import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ====================================
# STEP 1: LOAD DATASET
# ====================================

df = pd.read_csv("titanic.csv")

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA INFO =====")
print(df.info())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# ====================================
# STEP 2: REMOVE DUPLICATES
# ====================================

duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")

df.drop_duplicates(inplace=True)

# ====================================
# STEP 3: HANDLE MISSING VALUES
# ====================================

# Age -> Median
df["Age"].fillna(df["Age"].median(), inplace=True)

# Embarked -> Mode
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Cabin -> Too many missing values
if "Cabin" in df.columns:
    df.drop("Cabin", axis=1, inplace=True)

# ====================================
# STEP 4: OUTLIER REMOVAL
# ====================================

Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[
    (df["Fare"] >= lower) &
    (df["Fare"] <= upper)
]

print("\nDataset Shape After Cleaning:")
print(df.shape)

# ====================================
# STEP 5: SAVE CLEANED DATA
# ====================================

df.to_csv("cleaned_titanic.csv", index=False)

print("\nCleaned dataset saved as cleaned_titanic.csv")

# ====================================
# STEP 6: DESCRIPTIVE STATISTICS
# ====================================

print("\n===== STATISTICS =====")
print(df.describe())

# ====================================
# STEP 7: VISUALIZATION 1
# SURVIVAL COUNT
# ====================================

plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.savefig("survival_count.png")
plt.show()

# ====================================
# STEP 8: VISUALIZATION 2
# AGE DISTRIBUTION
# ====================================

plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.savefig("age_distribution.png")
plt.show()

# ====================================
# STEP 9: VISUALIZATION 3
# GENDER DISTRIBUTION
# ====================================

plt.figure(figsize=(6,4))
sns.countplot(x="Sex", data=df)
plt.title("Gender Distribution")
plt.savefig("gender_distribution.png")
plt.show()

# ====================================
# STEP 10: VISUALIZATION 4
# PASSENGER CLASS
# ====================================

plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", data=df)
plt.title("Passenger Class Distribution")
plt.savefig("passenger_class.png")
plt.show()

# ====================================
# STEP 11: VISUALIZATION 5
# SURVIVAL BY GENDER
# ====================================

plt.figure(figsize=(6,4))
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival By Gender")
plt.savefig("survival_by_gender.png")
plt.show()

# ====================================
# STEP 12: VISUALIZATION 6
# CORRELATION HEATMAP
# ====================================

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10,6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.savefig("heatmap.png")
plt.show()

# ====================================
# STEP 13: VISUALIZATION 7
# FARE DISTRIBUTION
# ====================================

plt.figure(figsize=(8,5))
sns.boxplot(x=df["Fare"])
plt.title("Fare Boxplot")
plt.savefig("fare_boxplot.png")
plt.show()

# ====================================
# STEP 14: INSIGHTS
# ====================================

print("\n===== KEY INSIGHTS =====")

survival_rate = df["Survived"].mean() * 100

print(f"Overall Survival Rate: {survival_rate:.2f}%")

print(
    "\nAverage Age:",
    round(df["Age"].mean(), 2)
)

print(
    "\nAverage Fare:",
    round(df["Fare"].mean(), 2)
)

print(
    "\nMost Common Embarkation Port:",
    df["Embarked"].mode()[0]
)

print("\nProject Completed Successfully!")
