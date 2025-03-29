# Import necessary libraries for data manipulation and visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = 'Electric_Vehicle_Population_Data.csv'  # Replace with your dataset's path
df = pd.read_csv(url)

# Set general style for the plots
sns.set(style="whitegrid")

# 1. Correlation Heatmap - only include numerical columns
plt.figure(figsize=(10, 8))
numerical_df = df.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
correlation_matrix = numerical_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", square=True)
plt.title("Correlation Heatmap")
plt.show()

# 2. Boxplot for Base MSRP by Electric Vehicle Type
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Electric Vehicle Type', y='Base MSRP', palette="Set2")
plt.title('Base MSRP by Electric Vehicle Type')
plt.xlabel('Electric Vehicle Type')
plt.ylabel('Base MSRP')
plt.xticks(rotation=45, ha='right')
plt.show()

# 3. Scatter Plot for Electric Range vs. Base MSRP
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Electric Range', y='Base MSRP', hue='Electric Vehicle Type', palette='viridis', alpha=0.6)
plt.title('Electric Range vs. Base MSRP')
plt.xlabel('Electric Range')
plt.ylabel('Base MSRP')
plt.legend(title='Electric Vehicle Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# 4. Histogram for Distribution of Electric Range
plt.figure(figsize=(10, 6))
sns.histplot(df['Electric Range'], bins=20, kde=True, color="skyblue")
plt.title('Distribution of Electric Range')
plt.xlabel('Electric Range')
plt.ylabel('Frequency')
plt.show()

# 5. Count Plot for Electric Vehicles by Type
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Electric Vehicle Type', palette="Set1")
plt.title('Count of Electric Vehicles by Type')
plt.xlabel('Electric Vehicle Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.show()

# 6. Pair Plot for Selected Features - again only select numerical columns
selected_features = ['Base MSRP', 'Electric Range', 'Model Year']
sns.pairplot(df[selected_features].select_dtypes(include=['float64', 'int64']), plot_kws={'alpha': 0.5})
plt.suptitle('Scatter Plot Matrix of Selected Features', y=1.02)
plt.show()