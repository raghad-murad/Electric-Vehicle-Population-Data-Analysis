import pandas as pd

# Load the dataset
url = 'Electric_Vehicle_Population_Data.csv'
df = pd.read_csv(url)

# Select numerical features
numerical_features = ['Model Year', 'Electric Range', 'Base MSRP']

# Calculate descriptive statistics
descriptive_stats = df[numerical_features].describe().transpose()
descriptive_stats['median'] = df[numerical_features].median()

# Print the descriptive statistics
print("Descriptive Statistics:")
print(descriptive_stats)

# If you want to save the statistics to a CSV file for further analysis or documentation
descriptive_stats.to_csv('Descriptive_Statistics.csv')