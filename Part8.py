import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
url = 'Electric_Vehicle_Population_Data.csv'
df = pd.read_csv(url)

# Select the relevant numerical features for correlation analysis
numerical_features = ['Postal Code', 'Model Year', 'Electric Range', 'Base MSRP', 'Legislative District', 'DOL Vehicle ID', '2020 Census Tract']

# Calculate the correlation matrix
correlation_matrix = df[numerical_features].corr()

print("Correlation Matrix:")
print(correlation_matrix)

# Create a heatmap to visualize correlations
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Numeric Features')
plt.show()