# Project Prediction

Dieses Repository enthält das Verzeichnis `Project_Prediction`, in dem zwei R-Skripte zur Vorhersage bzw. Prognose von Daten liegen.

## Content

- **Own_Running_Data_Analysis_and_Prediction.R**  
  *Description:* Loading and doing an EDA of my own running data and identifying the best days to run for me and predict future running finish time like a marathon.

- **RShiny_Running_Comparison_and_Prediction.R**  
  *Description:* Loading race finish times with its respective percentiles for the distance and gender. Creating a Shiny application where the user can check its race finish times in comparison to simiar runners in the category and also predicting race finish times based on a previous run using Riegel's formula.

## Voraussetzungen

- **R** (Version 4.1.1 oder höher)  
- Weitere benötigte R-Pakete:  
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

## Installation und Nutzung

1. **Repository klonen:**

   ```bash
   git clone https://github.com/DEIN_USERNAME/DEIN_REPOSITORY.git

## Run R-Script

- Open the scripts in RStudio or execute them directly in the R console:

 ```bash
   source("Own_Running_Data_Analysis_and_Prediction.R")
   source("RShiny_Running_Comparison_and_Prediction.R")
