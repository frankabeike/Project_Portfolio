# Project Prediction

This directory contains one R script and one python scrip for comparing and predicting finish times of runners.

## Content

- **Own_Running_Data_Analysis_and_Prediction.py**  
  *Description:* Loading and exploring my personal running data to identify optimal training days and predict future race finish times, such as for a marathon.

- **RShiny_Running_Comparison_and_Prediction.R**  
  *Description:* Loading race finish times along with their corresponding percentiles for each distance and gender category. The project includes a Shiny application that allows users to compare their finish times with those of similar runners, as well as to predict future race times based on previous performances using Riegel's formula.

## Prerequisites

- **Python** (Version 4.1.1 or higher)  
- Further required Python packages:
  - pandas
  - numpy
  - python-dateutil (for the dateutil.parser module)
  - matplotlib
  - seaborn
  - scikit-learn (for modules like LinearRegression, RandomForestRegressor, train_test_split, mean_squared_error, and GridSearchCV)
  - xgboost
  ```bash
   pip install pandas numpy python-dateutil matplotlib seaborn scikit-learn xgboost

- **R** (Version 4.1.1 or higher)  
- Further required R packages: 
  - pacman
  - dplyr
  - DT
  - sqldf
  - lubridate
  - stringr
  - shiny
  - shinyBS
  - shinythemes
  ```bash
  install.packages(c("pacman", "dplyr", "DT", "sqldf", "lubridate", "stringr", "shiny", "shinyBS", "shinythemes"))

## Run Python-Script

- **Open the script for example in VS Code:**
- 

    ```bash
   python Own_Running_Data_Analysis_and_Prediction.py

## Run R-Script

- **Open the scripts in RStudio or execute them directly in the R console:**
- Change the file path 

    ```bash
   source("RShiny_Running_Comparison_and_Prediction.R")
