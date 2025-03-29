import pandas as pd

# Load the dataset
url = 'Electric_Vehicle_Population_Data.csv'
df = pd.read_csv(url)  # Read the CSV file into a pandas DataFrame

# Display the first few rows of the dataset
print("Original DataFrame:")
print(df.head())

# Select categorical features to encode
categorical_features = ['Make', 'Model']

# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=categorical_features)

# Display the first few rows of the encoded DataFrame
print("\nOne-Hot Encoded DataFrame:")
print(df_encoded.head())

# Save the encoded DataFrame to a new CSV file (optional)
df_encoded.to_csv('Electric_Vehicle_Population_Data_Encoded.csv', index=False)