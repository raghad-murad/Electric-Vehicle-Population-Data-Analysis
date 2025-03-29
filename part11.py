import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = 'Electric_Vehicle_Population_Data.csv'
df = pd.read_csv(url)

# Ensure 'Model Year' is treated as a numeric feature
df['Model Year'] = pd.to_numeric(df['Model Year'], errors='coerce')

# 1. Analyze the EV adoption rates over time
ev_adoption = df.groupby('Model Year').size()

# Plotting the EV adoption rates over time
plt.figure(figsize=(12, 6))
ev_adoption.plot(kind='line', marker='o', color='teal')
plt.title('EV Adoption Rates Over Time')
plt.xlabel('Model Year')
plt.ylabel('Number of EVs')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Analyze the popularity of all EV models over time
model_popularity = df.groupby(['Model Year', 'Model']).size().unstack().fillna(0)

# Plotting the popularity of all EV models over time
plt.figure(figsize=(14, 8))
model_popularity.plot(kind='line', marker='o', colormap='tab20')
plt.title('Trends in Popularity of All EV Models Over Time')
plt.xlabel('Model Year')
plt.ylabel('Number of EVs')
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Spatial Distribution of EVs by City
# Plotting the popularity of top EV models over time as a bar chart 
colors = sns.color_palette('tab20', len(model_popularity.columns)) 
model_popularity.T.plot(kind='bar', stacked=True, figsize=(14, 8), color=colors) 
plt.title('Model Popularity Over the Years') 
plt.xlabel('Model Year') 
plt.ylabel('Number of Vehicles') 
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1) 
plt.xticks(rotation=45) 
plt.tight_layout() 
plt.show()