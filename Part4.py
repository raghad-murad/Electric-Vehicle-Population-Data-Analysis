import pandas as pd

# Load the dataset
url = 'Electric_Vehicle_Population_Data.csv'
df = pd.read_csv(url)  # Read the CSV file into a pandas DataFrame

# Display the first few rows of the original DataFrame
print("Original DataFrame:")
print(df.head())

# Select numerical features to normalize
numerical_features = ['Model Year', 'Electric Range', 'Base MSRP']

# Apply Min-Max Normalization
df_min_max_scaled = df.copy()
for feature in numerical_features:
    min_value = df[feature].min()
    max_value = df[feature].max()
    df_min_max_scaled[feature] = (df[feature] - min_value) / (max_value - min_value)

# Apply Standard Normalization
df_standard_scaled = df.copy()
for feature in numerical_features:
    mean_value = df[feature].mean()
    std_value = df[feature].std()
    df_standard_scaled[feature] = (df[feature] - mean_value) / std_value

# Display the first few rows of the normalized DataFrames
print("\nMin-Max Scaled DataFrame:")
print(df_min_max_scaled.head())

print("\nStandard Scaled DataFrame:")
print(df_standard_scaled.head())

# Save the normalized DataFrames to new CSV files (optional)
df_min_max_scaled.to_csv('Electric_Vehicle_Population_Data_MinMax_Scaled.csv', index=False)
df_standard_scaled.to_csv('Electric_Vehicle_Population_Data_Standard_Scaled.csv', index=False)