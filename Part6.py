import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = 'Electric_Vehicle_Population_Data.csv'
df = pd.read_csv(url)

# Analyze the spatial distribution of EVs by City
city_counts = df['City'].value_counts()

# Create a bar plot for the distribution of EVs by City
plt.figure(figsize=(12, 8))
city_counts.plot(kind='bar', color='teal')
plt.title('Spatial Distribution of EVs by City')
plt.xlabel('City')
plt.ylabel('Number of EVs')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Analyze the spatial distribution of EVs by County
county_counts = df['County'].value_counts()

# Create a bar plot for the distribution of EVs by County
plt.figure(figsize=(12, 8))
county_counts.plot(kind='bar', color='purple')
plt.title('Spatial Distribution of EVs by County')
plt.xlabel('County')
plt.ylabel('Number of EVs')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()