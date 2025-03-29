import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset:
url = 'Electric_Vehicle_Population_Data.csv'                          
df = pd.read_csv(url) # Read the CSV file into a pandas DataFrame

# Check for missing values in the DataFrame, by count the number of missing values in each column:
missing_values = df.isnull().sum()

# Document the frequency and distribution of missing values:
missing_frequency = pd.DataFrame({
    'Feature': missing_values.index,
    'Missing Values': missing_values.values,
    'Percentage': (missing_values.values / len(df)) * 100
})

# Print the summary of missing values frequency and percentage:
print("Missing Values Frequency and Percentage:")
print(missing_frequency)

# Plotting the distribution of missing values:
plt.figure(figsize=(12, 6))
sns.barplot(x='Feature', y='Missing Values', data=missing_frequency, hue='Feature', legend=False, palette='viridis')
plt.title('Missing Values Distribution', fontsize=16)
plt.xlabel('Features', fontsize=14)
plt.ylabel('Number of Missing Values', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12, rotation_mode='anchor')
plt.tight_layout()
plt.show()