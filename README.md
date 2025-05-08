# SpaceX Launch Data Analysis and Prediction

### Analyzing SpaceX Launch Data for Predictive Insights and Interactive Visualization

## Table of Contents
- [Overview](#overview)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Key Modules](#key-modules)
- [Results and Insights](#results-and-insights)
- [License](#license)

## Overview
This project aims to explore and analyze SpaceX launch data to gain insights into factors influencing launch success and to develop predictive models. We utilize data visualization, SQL-based exploratory data analysis, interactive dashboards, and machine learning to create a comprehensive tool for understanding and forecasting launch outcomes. This project is presented in the form of a cohesive, modular codebase, combining several key data science practices and libraries.

## Data Sources
The data for this project was gathered from the SpaceX API and includes detailed launch records such as:
- Launch sites, payloads, orbit types, and customer details
- Booster types, launch outcomes, and booster landing outcomes

Additional information was supplemented by historical data and external datasets to enrich the analysis.

## Project Structure
The project is organized into the following modules:

1. **Folium Map Visualization:** Creates an interactive map displaying the launch sites, providing geospatial insights on launch site distribution and regional patterns.
2. **SQL-Based Exploratory Data Analysis:** Uses SQL queries for an in-depth analysis of various launch parameters, revealing key patterns in success rates, booster landing rates, and payload influences.
3. **Machine Learning Prediction Model:** Builds and evaluates a machine learning model that predicts launch success probabilities based on historical data, analyzing key predictive factors.
4. **Interactive Dashboard:** Integrates data visualizations into a dashboard, enabling real-time data filtering and presentation of insights for users to explore SpaceX data interactively.

## Setup and Installation
To set up this project on your local machine, follow these instructions:

### Prerequisites
- Python 3.7+
- Jupyter Notebook
- Virtual environment (optional but recommended)

### Install Dependencies
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/spacex-data-analysis
   cd spacex-data-analysis
   ```
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project
1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open each notebook for step-by-step analysis and visualization, or run the combined script (`SpaceX_Combined_Script.py`) directly for an end-to-end analysis:
   ```bash
   python SpaceX_Combined_Script.py
   ```

## Usage
Each module is designed to run independently or as part of the full workflow. Here’s a quick guide for each component:

- **Folium Map Visualization:** Run the `spaceX_folium_map.ipynb` notebook to generate interactive maps of SpaceX launch sites.
- **SQL-Based Analysis:** Execute the `spaceX_exploratory_data_analysis_sql.ipynb` notebook for data queries and analysis using SQL.
- **Machine Learning Prediction:** Use `spacex_machine_learning_prediction.py` to train and evaluate models predicting launch success.
- **Dashboard:** Run the `spacex_dashboard.py` script to visualize data interactively with filters and data summaries.

## Key Modules

### 1. Folium Map Visualization
   - Uses Folium to create an interactive map showing SpaceX launch sites.
   - Enhances understanding of launch site distribution and geographical factors.

### 2. SQL-Based Exploratory Data Analysis
   - Utilizes SQL queries in Jupyter Notebook to analyze and extract insights from SpaceX data.
   - Key insights include the correlation of booster types, orbit, and payload with launch outcomes.

### 3. Machine Learning Prediction
   - Predicts the probability of a successful launch based on historical data.
   - Models trained include logistic regression, decision trees, and other classifiers, with performance evaluated on accuracy and recall metrics.

### 4. Interactive Dashboard
   - A dashboard that integrates visualizations for user-friendly data exploration.
   - Provides filters and live data views for launch success, booster types, and launch site comparisons.

## Results and Insights
This project reveals valuable insights into SpaceX’s launch patterns, including:
- **Launch Site Importance:** Certain launch sites show higher success rates, potentially due to location advantages.
- **Booster Reliability:** Specific booster versions have a significant impact on launch success rates.
- **Predictive Modeling:** Our machine learning models provide a reasonably accurate forecast of launch outcomes, supporting mission planning and risk assessment.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

