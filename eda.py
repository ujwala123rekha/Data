import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("Titanic-Dataset.csv")

print("="*50)
print("FIRST 5 ROWS")
print("="*50)
print(df.head())

print("\n")
print("="*50)
print("DATASET INFO")
print("="*50)
print(df.info())

print("\n")
print("="*50)
print("STATISTICAL SUMMARY")
print("="*50)
print(df.describe())

# ==========================
# CHECK MISSING VALUES
# ==========================

print("\n")
print("="*50)
print("MISSING VALUES")
print("="*50)
print(df.isnull().sum())

# Fill missing values

if "Age" in df.columns:
    df["Age"].fillna(df["Age"].median(), inplace=True)

if "Embarked" in df.columns:
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

if "Cabin" in df.columns:
    df["Cabin"].fillna("Unknown", inplace=True)

# ==========================
# SURVIVAL DISTRIBUTION
# ==========================

plt.figure(figsize=(6,4))

df["Survived"].value_counts().plot(
    kind="bar"
)

plt.title("Survival Distribution")
plt.xlabel("Survived")
plt.ylabel("Count")

plt.show()

# ==========================
# GENDER DISTRIBUTION
# ==========================

plt.figure(figsize=(6,4))

df["Sex"].value_counts().plot(
    kind="bar"
)

plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")

plt.show()

# ==========================
# AGE DISTRIBUTION
# ==========================

plt.figure(figsize=(8,5))

plt.hist(
    df["Age"],
    bins=20
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.show()

# ==========================
# SURVIVAL BY GENDER
# ==========================

survival_gender = pd.crosstab(
    df["Sex"],
    df["Survived"]
)

print("\n")
print("="*50)
print("SURVIVAL BY GENDER")
print("="*50)
print(survival_gender)

survival_gender.plot(
    kind="bar",
    figsize=(8,5)
)

plt.title("Survival by Gender")
plt.ylabel("Count")

plt.show()

# ==========================
# SURVIVAL BY PASSENGER CLASS
# ==========================

survival_class = pd.crosstab(
    df["Pclass"],
    df["Survived"]
)

print("\n")
print("="*50)
print("SURVIVAL BY CLASS")
print("="*50)
print(survival_class)

survival_class.plot(
    kind="bar",
    figsize=(8,5)
)

plt.title("Survival by Passenger Class")
plt.ylabel("Count")

plt.show()

# ==========================
# CORRELATION ANALYSIS
# ==========================

numeric_df = df.select_dtypes(
    include=np.number
)

correlation = numeric_df.corr()

print("\n")
print("="*50)
print("CORRELATION MATRIX")
print("="*50)
print(correlation)

plt.figure(figsize=(8,6))

plt.imshow(
    correlation,
    cmap="coolwarm"
)

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.show()

# ==========================
# TOP CORRELATIONS WITH TARGET
# ==========================

if "Survived" in correlation.columns:

    target_corr = correlation["Survived"].sort_values(
        ascending=False
    )

    print("\n")
    print("="*50)
    print("FACTORS AFFECTING SURVIVAL")
    print("="*50)
    print(target_corr)

# ==========================
# INSIGHTS REPORT
# ==========================

print("\n")
print("="*60)
print("FINAL INSIGHTS")
print("="*60)

print("""
1. Analyzed passenger demographics and survival patterns.

2. Investigated missing values and handled them.

3. Explored age, gender, and class distributions.

4. Performed correlation analysis on numerical features.

5. Identified important factors influencing survival.

6. Created visualizations for easier interpretation.

7. Generated statistical summaries for data understanding.
""")

print("\nEDA PROJECT COMPLETED SUCCESSFULLY!")
