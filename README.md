# Machine Learning Assignment 1: Electric Vehicle Population Data Analysis

This repository contains the implementation and report for **Assignment 1** in the Machine Learning course. The assignment focuses on analyzing the **Electric Vehicle Population Data** dataset provided by the State of Washington. The goal is to perform data preprocessing, conduct exploratory data analysis (EDA), and visualize insights from the dataset.

---

## 📚 Table of Contents

- [Machine Learning Assignment 1: Electric Vehicle Population Data Analysis](#machine-learning-assignment-1-electric-vehicle-population-data-analysis)
  - [📚 Table of Contents](#-table-of-contents)
  - [🌟 Overview](#-overview)
  - [📊 Dataset](#-dataset)
  - [🛠️ Implementation Details](#️-implementation-details)
  - [📁 Files in the Repository](#-files-in-the-repository)
    - [Main Files](#main-files)
    - [Data Files](#data-files)
    - [Script Files](#script-files)
  - [🚀 How to Run the Project](#-how-to-run-the-project)
    - [Prerequisites](#prerequisites)
    - [Steps to Run](#steps-to-run)
  - [📊 Results and Visualizations](#-results-and-visualizations)
  - [🤝 Contributions](#-contributions)
  - [📧 Contact](#-contact)
    - [Thank you for checking out this project! 🚀](#thank-you-for-checking-out-this-project-)

---

## 🌟 Overview

The objective of this assignment is to analyze the **Electric Vehicle Population Data** dataset to gain insights into electric vehicle registrations across Washington State. The analysis includes the following steps:

1. **Data Cleaning and Feature Engineering**:
   - Document missing values and apply strategies such as mean imputation, median imputation, and row removal.
   - Encode categorical features using one-hot encoding.
   - Normalize numerical features using Min-Max scaling and standardization.

2. **Exploratory Data Analysis (EDA)**:
   - Calculate descriptive statistics for numerical features.
   - Visualize spatial distributions of EVs by city and county.
   - Analyze the popularity of different EV models and types.
   - Investigate relationships between numeric features using correlation matrices.

3. **Visualization**:
   - Generate various plots to explore feature relationships, including correlation heatmaps, boxplots, scatter plots, histograms, count plots, and pair plots.
   - Create comparative visualizations across locations (cities and counties) to highlight EV distribution.

4. **Additional Analysis**:
   - Conduct temporal analysis to examine trends in EV adoption over time.

---

## 📊 Dataset

The dataset used in this project is titled **"Electric Vehicle Population Data"** and can be found on Data.gov: [https://catalog.data.gov/dataset/electric-vehicle-population-data](https://catalog.data.gov/dataset/electric-vehicle-population-data).

- **Source**: Provided by the State of Washington.
- **Description**: Contains information about battery electric vehicles (BEVs) and plug-in hybrid electric vehicles (PHEVs) registered through the Washington State Department of Licensing.
- **Columns**: Includes details such as VIN, county and city of registration, make and model, electric type, electric range, and vehicle model years.

---

## 🛠️ Implementation Details

The project is implemented using Python with the following libraries:
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib**: For basic plotting.
- **Seaborn**: For advanced data visualization.
- **Geopandas**: For spatial analysis and map plotting.

The implementation is divided into modular parts, each focusing on a specific task:

1. **Part 1: Document Missing Values**
   - Identifies and summarizes missing values in the dataset.
   - Visualizes the distribution of missing values across features.

2. **Part 2: Missing Value Strategies**
   - Applies strategies like mean imputation, median imputation, and row removal.
   - Compares the impact of these strategies on the dataset.

3. **Part 3: Feature Encoding**
   - Encodes categorical features using one-hot encoding.

4. **Part 4: Normalization**
   - Normalizes numerical features using Min-Max scaling and standardization.

5. **Part 5: Descriptive Statistics**
   - Calculates summary statistics for numerical features.

6. **Part 6: Spatial Distribution**
   - Visualizes the spatial distribution of EVs by city and county.
   - Performs geospatial analysis to plot EV distribution across regions.

7. **Part 7: Model Popularity**
   - Analyzes the popularity of different EV models and types.
   - Tracks trends in model popularity over time.

8. **Part 8: Investigate Relationships**
   - Computes and visualizes correlations between numeric features.

9. **Part 9: Data Exploration Visualizations**
   - Generates various plots to explore feature relationships.

10. **Part 10: Comparative Visualization**
    - Creates bar charts and stacked bar charts to compare EV distributions across cities and counties.

11. **Part 11: Temporal Analysis**
    - Examines trends in EV adoption over time.
    - Visualizes the popularity of EV models and types over the years.

---

## 📁 Files in the Repository

The repository contains the following files:

### Main Files
- **`assignment1_1212214.py`**: The main Python script that orchestrates all analysis tasks.
- **`ML_assignment_1.pdf`**: Detailed report explaining the dataset, methodology, results, and analysis.

### Data Files
- **`Electric_Vehicle_Population_Data.csv`**: Original dataset containing EV registration data.
- **`Descriptive_Statistics.csv`**: Summary statistics generated during EDA.
- **`Electric_Vehicle_Population_Data_Encoded.csv`**: Dataset after one-hot encoding.
- **`Electric_Vehicle_Population_Data_MinMax_Scaled.csv`**: Dataset after Min-Max normalization.
- **`Electric_Vehicle_Population_Data_Standard_Scaled.csv`**: Dataset after standardization.

### Script Files
- **`Part1.py` to `part11.py`**: Modular scripts for each part of the analysis.

---

## 🚀 How to Run the Project

### Prerequisites
- **Python**: Ensure Python is installed on your machine.
- **Libraries**: Install the required libraries using `pip`:
  ```bash
  pip install pandas numpy matplotlib seaborn geopandas
  ```

### Steps to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/raghad-murad/MachineLearningAssignment1.git
   ```

2. **Navigate to the Directory**
   ```bash
   cd MachineLearningAssignment1
   ```

3. **Run the Main Script**
   ```bash
   python assignment1_1212214.py
   ```

   Alternatively, you can run individual parts using the corresponding scripts:
   ```bash
   python Part1.py
   python Part2.py
   # ... and so on
   ```

4. **View Results**
   - Output CSV files will be generated in the directory.
   - Visualizations will be displayed in separate windows or saved as images.

---

## 📊 Results and Visualizations

The project generates various outputs, including:
- **CSV Files**: Summarized statistics and transformed datasets.
- **Visualizations**: Plots and maps showcasing spatial distributions, model popularity, correlations, and temporal trends.

Example visualizations include:
- Correlation heatmap of numeric features.
- Boxplot of Base MSRP by EV type.
- Scatter plot of Electric Range vs. Base MSRP.
- Histogram of Electric Range distribution.
- Count plot of EVs by type.
- Pair plot of selected features.
- Bar charts and stacked bar charts for EV distribution across cities and counties.
- Time series plots for EV adoption trends.

---

## 🤝 Contributions

If you'd like to contribute to this repository, feel free to:
1. Fork the repository.
2. Create a new branch for your changes.
3. Submit a pull request with a detailed explanation of your changes.

---

## 📧 Contact

If you have any questions or suggestions, feel free to reach out!

- **Email:** raghadmbuzia@gmail.com
- **LinkedIn:** [in/raghad-murad](http://linkedin.com/in/raghad-murad-02690433a)

---

### Thank you for checking out this project! 🚀