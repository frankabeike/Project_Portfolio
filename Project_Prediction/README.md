# Project Prediction

This directory contains two R scripts for comparing and predicting finish times of runners.

## Content

- **Own_Running_Data_Analysis_and_Prediction.R**  
  *Description:* Loading and exploring my personal running data to identify optimal training days and predict future race finish times, such as for a marathon.

- **RShiny_Running_Comparison_and_Prediction.R**  
  *Description:* Loading race finish times along with their corresponding percentiles for each distance and gender category. The project includes a Shiny application that allows users to compare their finish times with those of similar runners, as well as to predict future race times based on previous performances using Riegel's formula.

## Prerequisites

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
  - ggplot2
  - scales
  - readxl
  - corrplot  

## Run R-Script

- **Open the scripts in RStudio or execute them directly in the R console:**

 ```bash
   source("Own_Running_Data_Analysis_and_Prediction.R")
   source("RShiny_Running_Comparison_and_Prediction.R")
