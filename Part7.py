import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = 'Electric_Vehicle_Population_Data.csv'
df = pd.read_csv(url)

# --- 1. Top 10 Most Popular EV Models ---
# Analyze the popularity of different EV models
top_10_models = df['Model'].value_counts().head(10)

# Create a bar plot for the top 10 most popular EV models
plt.figure(figsize=(12, 8))
top_10_models.plot(kind='bar', color='teal')
plt.title('Top 10 Most Popular EV Models')
plt.xlabel('Model')
plt.ylabel('Number of EVs')
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend(title='Model', loc='upper left')
plt.show()

# --- 2. Trends in Popularity of Top EV Models Over Years ---
# Filter top models
top_models = df['Model'].value_counts().head(10).index
df_top_models = df[df['Model'].isin(top_models)]

# Trends in popularity over years
trends_over_years = df_top_models.groupby(['Model Year', 'Model']).size().unstack().fillna(0)

# Plotting
plt.figure(figsize=(14, 10))
trends_over_years.plot(kind='line', marker='o', colormap='tab10')
plt.title('Trends in Popularity of Top EV Models Over Years')
plt.xlabel('Model Year')
plt.ylabel('Number of EVs')
plt.legend(title='Model', loc='upper left')
plt.tight_layout()
plt.show()

# --- 3. Distribution of Electric Vehicle Types ---
# Analyze the distribution of EV types
ev_type_distribution = df['Electric Vehicle Type'].value_counts()

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Pie chart for the distribution of electric vehicle types
axs[0].pie(ev_type_distribution, labels=ev_type_distribution.index, autopct='%1.1f%%', colors=['gold', 'lightblue'], startangle=140)
axs[0].set_title('Distribution of Electric Vehicle Types (Pie Chart)')

# Bar chart for the distribution of electric vehicle types
axs[1].barh(ev_type_distribution.index, ev_type_distribution.values, color=['gold', 'lightblue'])
axs[1].set_xlabel('Number of Vehicles')
axs[1].set_title('Distribution of Electric Vehicle Types (Bar Chart)')

# Display the plots
plt.tight_layout()
plt.show()

# --- 4. Trends in Electric Vehicle Types Over Years ---
# Trends in EV types over years
trends_ev_types = df.groupby(['Model Year', 'Electric Vehicle Type']).size().unstack().fillna(0)

# Plotting
plt.figure(figsize=(14, 10))
trends_ev_types.plot(kind='line', marker='o', colormap='tab10')
plt.title('Trends in Electric Vehicle Types Over Years')
plt.xlabel('Model Year')
plt.ylabel('Number of EVs')
plt.legend(title='Electric Vehicle Type', loc='upper left')
plt.tight_layout()
plt.show()