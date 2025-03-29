import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from a local file
file_path = 'Electric_Vehicle_Population_Data.csv'  # Update this with your local file path
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("Initial Dataset:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Strategy 1: Mean Imputation
df_mean_imputed = df.copy()
for column in df_mean_imputed.select_dtypes(include=[np.number]).columns:
    df_mean_imputed[column].fillna(df_mean_imputed[column].mean(), inplace=True)

# Strategy 2: Median Imputation
df_median_imputed = df.copy()
for column in df_median_imputed.select_dtypes(include=[np.number]).columns:
    df_median_imputed[column].fillna(df_median_imputed[column].median(), inplace=True)

# Strategy 3: Dropping Rows
df_dropped = df.dropna()

# Compare the number of rows in each dataset
print("\nNumber of rows after each strategy:")
print(f"Original: {len(df)}")
print(f"Mean Imputed: {len(df_mean_imputed)}")
print(f"Median Imputed: {len(df_median_imputed)}")
print(f"Dropped Rows: {len(df_dropped)}")

# Exploratory Data Analysis (EDA)
# Example: Distribution of Electric Range
plt.figure(figsize=(12, 6))

# Plot for Original Data
plt.subplot(2, 2, 1)
sns.histplot(df['Electric Range'], bins=30, kde=True)
plt.title('Original Electric Range Distribution')

# Plot for Mean Imputed Data
plt.subplot(2, 2, 2)
sns.histplot(df_mean_imputed['Electric Range'], bins=30, kde=True)
plt.title('Mean Imputed Electric Range Distribution')

# Plot for Median Imputed Data
plt.subplot(2, 2, 3)
sns.histplot(df_median_imputed['Electric Range'], bins=30, kde=True)
plt.title('Median Imputed Electric Range Distribution')

# Plot for Dropped Rows Data
plt.subplot(2, 2, 4)
sns.histplot(df_dropped['Electric Range'], bins=30, kde=True)
plt.title('Dropped Rows Electric Range Distribution')

plt.tight_layout()
plt.show()

# Additional EDA: Count of Electric Types
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Electric Type')
plt.title('Count of Electric Types')
plt.xticks(rotation=45)
plt.show()

# Summary Statistics
print("\nSummary Statistics for Original Data:")
print(df.describe())
print("\nSummary Statistics for Mean Imputed Data:")
print(df_mean_imputed.describe())
print("\nSummary Statistics for Median Imputed Data:")
print(df_median_imputed.describe())
print("\nSummary Statistics for Dropped Rows Data:")
print(df_dropped.describe())