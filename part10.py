# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = 'Electric_Vehicle_Population_Data.csv'  # Replace with your dataset's path
df = pd.read_csv(url)

# Set general style for the plots
sns.set(style="whitegrid")

# 1. Bar Chart for EV Distribution Across Cities
plt.figure(figsize=(14, 8))
top_cities = df['City'].value_counts().nlargest(20)  # Select top 10 cities with the most EVs
sns.barplot(x=top_cities.index, y=top_cities.values, palette="Blues_d")
plt.title('Top 10 Cities with Most Electric Vehicles')
plt.xlabel('City')
plt.ylabel('Number of Electric Vehicles')
plt.xticks(rotation=45, ha='right')
plt.show()

# 2. Bar Chart for EV Distribution Across Counties
plt.figure(figsize=(14, 8))
top_counties = df['County'].value_counts().nlargest(20)  # Select top 10 counties with the most EVs
sns.barplot(x=top_counties.index, y=top_counties.values, palette="Greens_d")
plt.title('Top 10 Counties with Most Electric Vehicles')
plt.xlabel('County')
plt.ylabel('Number of Electric Vehicles')
plt.xticks(rotation=45, ha='right')
plt.show()

# 1. Stacked Bar Chart for EV Distribution Across All Cities by Vehicle Type
plt.figure(figsize=(14, 20))  # Adjust height for better readability if there are many cities
city_type_data = df.groupby(['City', 'Electric Vehicle Type']).size().unstack(fill_value=0)
city_type_data.plot(kind='bar', stacked=True, figsize=(14, 20), colormap="viridis")
plt.title('Distribution of Electric Vehicle Types Across All Cities')
plt.xlabel('City')
plt.ylabel('Number of Electric Vehicles')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Electric Vehicle Type', bbox_to_anchor=(1.05, 1), loc='upper left')  # Adjust legend position
plt.tight_layout()
plt.show()

# 2. Stacked Bar Chart for EV Distribution Across All Counties by Vehicle Type
plt.figure(figsize=(14, 20))  # Adjust height for better readability if there are many counties
county_type_data = df.groupby(['County', 'Electric Vehicle Type']).size().unstack(fill_value=0)
county_type_data.plot(kind='bar', stacked=True, figsize=(14, 20), colormap="plasma")
plt.title('Distribution of Electric Vehicle Types Across All Counties')
plt.xlabel('County')
plt.ylabel('Number of Electric Vehicles')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Electric Vehicle Type', bbox_to_anchor=(1.05, 1), loc='upper left')  # Adjust legend position
plt.tight_layout()
plt.show()