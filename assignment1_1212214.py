'''
Assignment 1: 

Overview:
     The aim of this assignment is to analyze and gain insights from the "Electric Vehicle Population Data" dataset, provided by the State of Washington.
     This dataset contains details on registered electric vehicles (EVs) across the state, including make, model, electric range, model year, and registration -
     locations. The analysis involves comprehensive data preprocessing and exploratory data analysis (EDA) to uncover trends and relationships within the data.
     Key steps include documenting and addressing missing values by evaluating multiple strategies (mean imputation, median imputation, and row removal) and 
     automatically selecting the best approach to maintain data integrity. Categorical features are encoded using one-hot encoding, and numerical features are
     normalized to ensure consistency for analysis. EDA is conducted through descriptive statistics, spatial distribution visualizations, model popularity trends,
     and correlation analysis, supported by various plots to illustrate data patterns and relationships. The code ultimately provides a complete workflow, 
     from data cleaning and preprocessing to detailed analysis and visualization, to understand the characteristics and trends in EV registrations across Washington State.

Name: Raghad Murad Buzia
ID #: 1212214
sec: 3
'''

# Import necessary libraries:
import pandas as pd                    # for data manipulation
import numpy as np                     # for numerical operations
import matplotlib.pyplot as plt        # for data visualization
import seaborn as sns                  # for data visualization
import geopandas as gpd                # for spatial analysis and map plotting

###############################################################################

#                    Data Cleaning and Feature Engineering                    #
 
###############################################################################

# Part 1: Document Missing Values
'''
This function explore into missing values within a dataset by counting the missing values in each column and calculating their percentage.
It prints a comprehensive summary of the missing values and then visualizes the data by plotting a bar chart that highlights the distribution
of missing values across different features. This helps to identify which columns have missing data.
'''
def document_missing_values(df):
    # Check for missing values in the DataFrame, by count the number of missing values in each column:
    missing_values = df.isnull().sum()

    # Document the frequency and distribution of missing values:
    missing_frequency = pd.DataFrame({
        'Feature': missing_values.index,
        'Missing Values': missing_values.values,
        'Percentage': (missing_values.values / len(df)) * 100
    })

    # Print the summary of missing values frequency and percentage:
    print("\n\n\nMissing Values Frequency and Percentage:")
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
    
# Part 2: Missing Value Strategies
'''
This function employs several approaches to manage missing values and evaluates the results.
1. Mean Imputation: Fills missing values in numeric columns with the mean of each column.
2. Median Imputation: Fills missing values in numeric columns with the median.
3. Drop Rows: Removes rows with any missing values.
The function then calculates summary statistics for each strategy to enable comparison and 
visualizes the impact of each method on a specific numerical feature, such as "Electric Range".
'''
def apply_missing_value_strategies(df):
    # Check for missing values in the DataFrame:
    missing_values = df.isnull().sum()
    print("\n\n\nMissing Values Frequency and Percentage:")
    print(missing_values)
    print("\n")

    # Define a function to compare missing value strategies:
    def compare_missing_value_strategies(df):
        strategies = {
            'Original': df.copy(),
            'Mean Imputation': df.copy(),
            'Median Imputation': df.copy(),
            'Drop Rows': df.copy()
        }

        # Mean Imputation:
        for column in df.select_dtypes(include=[np.number]).columns:
            strategies['Mean Imputation'][column] = strategies['Mean Imputation'][column].fillna(strategies['Mean Imputation'][column].mean())

        # Median Imputation:
        for column in df.select_dtypes(include=[np.number]).columns:
            strategies['Median Imputation'][column] = strategies['Median Imputation'][column].fillna(strategies['Median Imputation'][column].median())

        # Drop Rows:
        strategies['Drop Rows'] = strategies['Drop Rows'].dropna()

        # Summary statistics for comparison:
        summary_stats = {}
        for strategy, df_strategy in strategies.items():
            summary_stats[strategy] = df_strategy.describe()

        return strategies, summary_stats

    # Apply the strategies and get the summary statistics:
    strategies, summary_stats = compare_missing_value_strategies(df)

    # Print summary statistics for each strategy:
    for strategy, stats in summary_stats.items():
        print(f"\nSummary Statistics for {strategy}:")
        print(stats)

    # Plotting the impact of missing value strategies on a numerical feature (e.g., 'Electric Range'):
    plt.figure(figsize=(12, 8))

    for strategy, df_strategy in strategies.items():
        sns.histplot(df_strategy['Electric Range'].dropna(), kde=True, label=strategy, element="step")

    plt.title('Impact of Missing Value Strategies on Electric Range')
    plt.xlabel('Electric Range')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Part 3: Feature Encoding
'''
This function conducts encoding on categorical features by first selecting specific columns, namely Make and Model. 
It then employs one-hot encoding to convert these columns into numeric format. Additionally, it save the encoded DataFrame 
as a CSV file.
'''
def feature_encoding(df):
    # Display the first few rows of the dataset:
    print("\n\n\nOriginal DataFrame:")
    print(df.head())

    # Select categorical features to encode:
    categorical_features = ['Make', 'Model']

    # One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=categorical_features)

    # Display the first few rows of the encoded DataFrame:
    print("\nOne-Hot Encoded DataFrame:")
    print(df_encoded.head())

    # Save the encoded DataFrame to a new CSV file:
    df_encoded.to_csv('Electric_Vehicle_Population_Data_Encoded.csv', index=False)

# Part 4: Normalization
'''
This function normalizes numerical features to ensure uniform scaling by utilizing Min-Max Normalization, which scales valuesbetween 0 and 1,
and Standard Normalization, which centers values around the mean with unit variance. It also save each normalized DataFrame, both min-max and
standard scaled, as separate CSV files.
'''
def normalization(df):
    # Display the first few rows of the original DataFrame:
    print("\n\n\nOriginal DataFrame:")
    print(df.head())
    print("\n")

    # Select numerical features to normalize:
    numerical_features = ['Model Year', 'Electric Range', 'Base MSRP']

    # Apply Min-Max Normalization:
    df_min_max_scaled = df.copy()
    for feature in numerical_features:
        min_value = df[feature].min()
        max_value = df[feature].max()
        df_min_max_scaled[feature] = (df[feature] - min_value) / (max_value - min_value)

    # Apply Standard Normalization:
    df_standard_scaled = df.copy()
    for feature in numerical_features:
        mean_value = df[feature].mean()
        std_value = df[feature].std()
        df_standard_scaled[feature] = (df[feature] - mean_value) / std_value

    # Display the first few rows of the normalized DataFrames:
    print("\nMin-Max Scaled DataFrame:")
    print(df_min_max_scaled.head())

    print("\nStandard Scaled DataFrame:")
    print(df_standard_scaled.head())

    # Save the normalized DataFrames to new CSV files:
    df_min_max_scaled.to_csv('Electric_Vehicle_Population_Data_MinMax_Scaled.csv', index=False)
    df_standard_scaled.to_csv('Electric_Vehicle_Population_Data_Standard_Scaled.csv', index=False)

###############################################################################

#                          Exploratory Data Analysis                          #
 
###############################################################################

# Part 5: Descriptive Statistics
'''
This function calculates and displays descriptive statistics for numerical features by computing standard statistics, such as the mean, minimum,
maximum, standard deviation, and the median. It prints these statistics and save them to a CSV file.
'''
def descriptive_statistics(df):
    # Select numerical features:
    numerical_features = ['Model Year', 'Electric Range', 'Base MSRP']

    # Calculate descriptive statistics:
    descriptive_stats = df[numerical_features].describe().transpose()
    descriptive_stats['median'] = df[numerical_features].median()

    # Print the descriptive statistics:
    print("\n\n\nDescriptive Statistics:")
    print(descriptive_stats)
    print("\n")

    # Save the statistics to a CSV file:
    descriptive_stats.to_csv('Descriptive_Statistics.csv')

# Part 6: Spatial Distribution
'''
This function visualizes the spatial distribution of electric vehicles by creating bar charts to display the number of EVs registered in various cities
and counties. It also performs geospatial analysis by loading a GeoJSON file containing region boundaries, merging this data with EV information by city,
and then plotting a map to illustrate the distribution of EVs across different regions.
'''
def spatial_distribution_visualization(df):
    # Analyze the spatial distribution of EVs by City:
    city_counts = df['City'].value_counts()

    # Create a bar plot for the distribution of EVs by City:
    plt.figure(figsize=(12, 8))
    city_counts.plot(kind='bar', color='teal')
    plt.title('Spatial Distribution of EVs by City')
    plt.xlabel('City')
    plt.ylabel('Number of EVs')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # Analyze the spatial distribution of EVs by County:
    county_counts = df['County'].value_counts()

    # Create a bar plot for the distribution of EVs by County:
    plt.figure(figsize=(12, 8))
    county_counts.plot(kind='bar', color='purple')
    plt.title('Spatial Distribution of EVs by County')
    plt.xlabel('County')
    plt.ylabel('Number of EVs')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # Load GeoJSON file for geographic boundaries:
    gdf = gpd.read_file("gz_2010_us_040_00_5m.json")

    # Calculate the number of vehicles per area and merge it with GeoDataFrame:
    city_counts = df.groupby('City').size().reset_index(name='Number_of_EVs')
    gdf = gdf.merge(city_counts, left_on="NAME", right_on="City", how="left") 

    # Set missing values ​​to 0 for areas with no data:
    gdf['Number_of_EVs'] = gdf['Number_of_EVs'].fillna(0)

    # Plot the map:
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    gdf.plot(column='Number_of_EVs', cmap='viridis', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=gdf['Number_of_EVs'].max()))
    sm._A = []
    plt.title("Electric Vehicle Distribution Across Regions")
    plt.show()

# Part 7: Model Popularity
'''
This function delves into the popularity of various EV models by plotting the top 20 most popular EV models in a bar chart, showcasing how their popularity
has shifted over the years. It illustrates the distribution of different EV types, such as BEV versus PHEV, using both pie and bar charts and tracks the trend
of each EV type's popularity over time through detailed plots.
'''
def model_popularity_analysis(df):
    # 1. Top 20 Most Popular EV Models:

    # Analyze the popularity of different EV models:
    top_20_models = df['Model'].value_counts().head(20)

    # Create a bar plot for the top 20 most popular EV models:
    plt.figure(figsize=(12, 8))
    top_20_models.plot(kind='bar', color='teal')
    plt.title('Top 20 Most Popular EV Models')
    plt.xlabel('Model')
    plt.ylabel('Number of EVs')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend(title='Model', loc='upper left')
    plt.show()

    # 2. Trends in Popularity of Top EV Models Over Years:
    # Filter top models:
    top_models = df['Model'].value_counts().head(20).index
    df_top_models = df[df['Model'].isin(top_models)]

    # Trends in popularity over years:
    trends_over_years = df_top_models.groupby(['Model Year', 'Model']).size().unstack().fillna(0)

    # Plotting:
    plt.figure(figsize=(14, 10))
    trends_over_years.plot(kind='line', marker='o', colormap='tab10')
    plt.title('Trends in Popularity of Top EV Models Over Years')
    plt.xlabel('Model Year')
    plt.ylabel('Number of EVs')
    plt.legend(title='Model', loc='upper left')
    plt.tight_layout()
    plt.show()

    # 3. Distribution of Electric Vehicle Types:

    # Analyze the distribution of EV types:
    ev_type_distribution = df['Electric Vehicle Type'].value_counts()

    # Create a figure with two subplots:
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    # Pie chart for the distribution of electric vehicle types:
    axs[0].pie(ev_type_distribution, labels=ev_type_distribution.index, autopct='%1.1f%%', colors=['gold', 'lightblue'], startangle=140)
    axs[0].set_title('Distribution of Electric Vehicle Types (Pie Chart)')
    # Bar chart for the distribution of electric vehicle types:
    axs[1].barh(ev_type_distribution.index, ev_type_distribution.values, color=['gold', 'lightblue'])
    axs[1].set_xlabel('Number of Vehicles')
    axs[1].set_title('Distribution of Electric Vehicle Types (Bar Chart)')
    # Display the plots:
    plt.tight_layout()
    plt.show()

    # 4. Trends in Electric Vehicle Types Over Years:

    # Trends in EV types over years:
    trends_ev_types = df.groupby(['Model Year', 'Electric Vehicle Type']).size().unstack().fillna(0)

    # Plotting:
    plt.figure(figsize=(14, 10))
    trends_ev_types.plot(kind='line', marker='o', colormap='tab10')
    plt.title('Trends in Electric Vehicle Types Over Years')
    plt.xlabel('Model Year')
    plt.ylabel('Number of EVs')
    plt.legend(title='Electric Vehicle Type', loc='upper left')
    plt.tight_layout()
    plt.show()

# Part 8: Investigate the relationship between every pair of numeric features
'''
This function explores relationships between numeric features by calculating and printing a correlation matrix for selected numeric features. 
It then visualizes the matrix with a heatmap to make correlations easier to interpret.
'''
def correlation_investigation(df):
    # Select the relevant numerical features for correlation analysis:
    numerical_features = ['Postal Code', 'Model Year', 'Electric Range', 'Base MSRP', 'Legislative District', 'DOL Vehicle ID', '2020 Census Tract']

    # Calculate the correlation matrix:
    correlation_matrix = df[numerical_features].corr()

    print("\n\n\nCorrelation Matrix:")
    print(correlation_matrix)
    print("\n")

    # Create a heatmap to visualize correlations:
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Numeric Features')
    plt.show()

###############################################################################

#                                Visualization                                #
 
###############################################################################

# Part 9: Data Exploration Visualizations
'''
This function generates various plots to examine feature relationships, including a Correlation Heatmap for numeric features, a Boxplot 
of Base MSRP by EV Type, a Scatter Plot of Electric Range versus Base MSRP, a Histogram depicting the distribution of Electric Range,
 a Count Plot for the number of EVs by type, and a Pair Plot to visualize pairwise relationships among selected numeric features.
'''
def data_exploration_visualizations(df):
    # Set general style for the plots:
    sns.set(style="whitegrid")

    # 1. Correlation Heatmap - only include numerical columns:
    plt.figure(figsize=(10, 8))
    numerical_df = df.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
    correlation_matrix = numerical_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap")
    plt.show()

    # 2. Boxplot for Base MSRP by Electric Vehicle Type:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Electric Vehicle Type', y='Base MSRP', palette="Set2")
    plt.title('Base MSRP by Electric Vehicle Type')
    plt.xlabel('Electric Vehicle Type')
    plt.ylabel('Base MSRP')
    plt.xticks(rotation=45, ha='right')
    plt.show()

    # 3. Scatter Plot for Electric Range vs. Base MSRP:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Electric Range', y='Base MSRP', hue='Electric Vehicle Type', palette='viridis', alpha=0.6)
    plt.title('Electric Range vs. Base MSRP')
    plt.xlabel('Electric Range')
    plt.ylabel('Base MSRP')
    plt.legend(title='Electric Vehicle Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    # 4. Histogram for Distribution of Electric Range:
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Electric Range'], bins=20, kde=True, color="skyblue")
    plt.title('Distribution of Electric Range')
    plt.xlabel('Electric Range')
    plt.ylabel('Frequency')
    plt.show()

    # 5. Count Plot for Electric Vehicles by Type:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Electric Vehicle Type', palette="Set1")
    plt.title('Count of Electric Vehicles by Type')
    plt.xlabel('Electric Vehicle Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.show()

    # 6. Pair Plot for Selected Features - again only select numerical columns:
    selected_features = ['Base MSRP', 'Electric Range', 'Model Year']
    sns.pairplot(df[selected_features].select_dtypes(include=['float64', 'int64']), plot_kws={'alpha': 0.5})
    plt.suptitle('Scatter Plot Matrix of Selected Features', y=1.02)
    plt.show()

# Part 10: Comparative Visualization
'''
This function generates comparative visualizations across various locations by creating bar charts that highlight the top cities and counties with
the highest number of EVs. Additionally, it produces stacked bar charts to visualize the distribution of different EV types within each city and county,
thus facilitating comparisons of EV types across various locations.
'''
def comparative_visualization(df):
    # Set general style for the plots:
    sns.set(style="whitegrid")

    # 1. Bar Chart for EV Distribution Across Cities:
    plt.figure(figsize=(14, 8))
    top_cities = df['City'].value_counts().nlargest(20)  # Select top 20 cities with the most EVs
    sns.barplot(x=top_cities.index, y=top_cities.values, palette="Blues_d")
    plt.title('Top 20 Cities with Most Electric Vehicles')
    plt.xlabel('City')
    plt.ylabel('Number of Electric Vehicles')
    plt.xticks(rotation=45, ha='right')
    plt.show()

    # 2. Bar Chart for EV Distribution Across Counties:
    plt.figure(figsize=(14, 8))
    top_counties = df['County'].value_counts().nlargest(20)  # Select top 20 counties with the most EVs
    sns.barplot(x=top_counties.index, y=top_counties.values, palette="Greens_d")
    plt.title('Top 20 Counties with Most Electric Vehicles')
    plt.xlabel('County')
    plt.ylabel('Number of Electric Vehicles')
    plt.xticks(rotation=45, ha='right')
    plt.show()

    # 3. Stacked Bar Chart for EV Distribution Across All Cities by Vehicle Type:
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

    # 4. Stacked Bar Chart for EV Distribution Across All Counties by Vehicle Type:
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

###############################################################################

#                             Additional Analysis                             #
 
###############################################################################

# Part 11: Temporal Analysis
'''
This function examines trends in EV adoption over time by plotting a time series that displays the number of EVs registered each year, showcasing the popularity
trends of different EV models through visualizations of registration numbers over time. Additionally, it creates a stacked bar chart to illustrate the evolving
popularity of specific models over the years.
'''
def temporal_analysis(df):
    # Ensure 'Model Year' is treated as a numeric feature:
    df['Model Year'] = pd.to_numeric(df['Model Year'], errors='coerce')

    # 1. Analyze the EV adoption rates over time:
    ev_adoption = df.groupby('Model Year').size()

    # Plotting the EV adoption rates over time:
    plt.figure(figsize=(12, 6))
    ev_adoption.plot(kind='line', marker='o', color='teal')
    plt.title('EV Adoption Rates Over Time')
    plt.xlabel('Model Year')
    plt.ylabel('Number of EVs')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2. Analyze the popularity of all EV models over time:
    model_popularity = df.groupby(['Model Year', 'Model']).size().unstack().fillna(0)

    # Plotting the popularity of all EV models over time:
    plt.figure(figsize=(14, 8))
    model_popularity.plot(kind='line', marker='o', colormap='tab20')
    plt.title('Trends in Popularity of All EV Models Over Time')
    plt.xlabel('Model Year')
    plt.ylabel('Number of EVs')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3. Spatial Distribution of EVs by City:

    # Plotting the popularity of top EV models over time as a bar chart:
    colors = sns.color_palette('tab20', len(model_popularity.columns)) 
    model_popularity.T.plot(kind='bar', stacked=True, figsize=(14, 8), color=colors) 
    plt.title('Model Popularity Over the Years') 
    plt.xlabel('Model Year') 
    plt.ylabel('Number of Vehicles') 
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1) 
    plt.xticks(rotation=45) 
    plt.tight_layout() 
    plt.show()

################################################################################

#                                Main Functions                                #
 
################################################################################

'''
This function displays a menu of analysis options for the user, each corresponding to one of the main parts of the analysis.
'''
def display_menu():
        print("Electric Vehicle Population Data Analysis")
        print("1. Document Missing Values")
        print("2. Missing Value Strategies")
        print("3. Feature Encoding")
        print("4. Normalization")
        print("5. Descriptive Statistics")
        print("6. Spatial Distribution")
        print("7. Model Popularity")
        print("8. Investigate Relationships")
        print("9. Data Exploration Visualizations")
        print("10. Comparative Visualization")
        print("11. Temporal Analysis")
        print("\nNote: if you want to exit, press 0.\n")

# Main function
'''
The process begins with data loading, where the dataset is read into a DataFrame. The user interface then continuously prompts the user to choose an analysis option
from the menu, calling the appropriate function based on the user's selection. This loop continues until the user decides to exit by entering "0".
'''
def main():

    #display the menue and ask the user to choose the part he need to execute:
    display_menu()
    choice = input("Choose the part you want to execute (1-11): ")

    # Load the dataset:
    url = 'Electric_Vehicle_Population_Data.csv' 
    # Read the CSV file into a pandas DataFrame:
    df = pd.read_csv(url)
    
    while(choice != '0'):

        match choice:
            case '1':
                document_missing_values(df)
            case '2':
                apply_missing_value_strategies(df)
            case '3':
                feature_encoding(df)
            case '4':
                normalization(df)
            case '5':
                descriptive_statistics(df)
            case '6':
                spatial_distribution_visualization(df)
            case '7':
                model_popularity_analysis(df)
            case '8':
                correlation_investigation(df)
            case '9':
                data_exploration_visualizations(df)
            case '10':
                comparative_visualization(df)
            case '11':
                temporal_analysis(df)
            case _:
                print("Invalid choice. Please select a number between 1 and 10.")

        #display the menue and ask the user to choose the part he need to execute:
        display_menu()
        choice = input("Choose the part you want to execute (1-11): ")

    print("Exiting program :)")

# Run the main function
if __name__ == "__main__":
    main()