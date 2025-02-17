# install pacman
install.packages('pacman')
# load needed libraries
library(pacman)
pacman:: p_load(dplyr,DT,sqldf,lubridate, stringr, shiny)

# Load dataset for running times data
file_path_running <- "your/file/path.csv"
# Read the CSV file, separating columns by semicolons and filling in missing data
data_running <- read.csv(file_path_running, sep = ";", fill = TRUE)
# Correct column name for 'Gender' column
colnames(data_running)[colnames(data_running) == "ï..Gender"] <- "Gender"
# Convert Distance to numeric format, replacing commas with periods
data_running$Distance <- as.numeric(gsub(",", ".", data_running$Distance))
# Convert 'Time' column to seconds using the 'period_to_seconds' function and 'hms'
data_running$Time_Seconds <- period_to_seconds(hms(data_running$Time))
# Define labels for different running distances
distance_labels <- c("5K" = 5, "10K" = 10, "Half Marathon" = 21.1, "Marathon" = 42.195)
# Riegel's formula for predicting long-distance times based on short-distance performance
riegel_predict <- function(TimeShortDistance, ShortDistance, LongDistance, exponent = 1.06) {
  TimeLongDistance <- TimeShortDistance * (LongDistance / ShortDistance)^exponent
  return(TimeLongDistance)
}


# Define UI for Shiny application
ui <- fluidPage(
  titlePanel("Running Percentile Calculator"), # Title of the application
  
  # Sidebar layout with input controls for selecting distance, gender, and inputting time
  sidebarLayout(
    sidebarPanel(
      selectInput("distance", "Select Distance", choices = distance_labels), # Dropdown menu for selecting running distance
      selectInput("gender", "Select Gender", choices = c("All", "Female", "Male")), # Dropdown menu for selecting gender
      numericInput("hours", "Hours", value = 0, min = 0), # Numeric input for entering hours
      numericInput("minutes", "Minutes", value = 0, min = 0, max = 59), # Numeric input for entering minutes
      numericInput("seconds", "Seconds", value = 0, min = 0, max = 59), # Numeric input for entering seconds for the time
      actionButton("calculate", "Calculate Percentile") # Action button to calculate the percentile
    ),
    # Main panel for displaying results and outputs
    mainPanel(
      verbatimTextOutput("result"), # Display percentile result
      verbatimTextOutput("summary_text"), # Display Summary of Finish Time 
      textOutput("text"), # Display title of table
      tableOutput("table") # Display table with percentile and times
    )
  ),
  # Second sidebar layout for predicting time using Riegel's formula
  sidebarLayout(
    sidebarPanel(
      numericInput("shortest_distance", "Enter Shortest Distance (km)", value = 5, min = 1), # Numeric inputs for entering shortest distance
      numericInput("shortest_hours", "Shortest Distance Hours", value = 0, min = 0), # Numeric inputs for entering hours
      numericInput("shortest_minutes", "Shortest Distance Minutes", value = 0, min = 0, max = 59), # Numeric inputs for entering minutes
      numericInput("shortest_seconds", "Shortest Distance Seconds", value = 0, min = 0, max = 59), # Numeric inputs for entering seconds
      selectInput("predicted_distance", "Select Predicted Distance", choices = distance_labels), # Dropdown menu for selecting the predicted distance
      actionButton("predict", "Predict Time") # Action button to predict time based on Riegel's formula
    ),
    # Main panel for displaying predicted time and percentile table
    mainPanel(
      verbatimTextOutput("predicted_time"), # Display predicted time
      textOutput("text_predict"), # Display title of table
      tableOutput("table_predict") # Display table with percentile and times
    )
  )
)

# Define Server logic
server <- function(input, output) {
  
  # Function to get the ordinal suffix (st, nd, rd, th) for numbers (e.g., 1st, 2nd, 3rd,...,10th)
  get_ordinal_suffix <- function(n) {
    ifelse(n %% 10 == 1 & n %% 100 != 11, "st",
           ifelse(n %% 10 == 2 & n %% 100 != 12, "nd",
                  ifelse(n %% 10 == 3 & n %% 100 != 13, "rd", "th")))
  }
  
  # Event handler for the "Calculate Percentile" button
  observeEvent(input$calculate, {
    
    # rename the dataset
    run_data <- data_running
    # Filter the dataset based on the selected gender
    run_data <- run_data %>% filter(Gender == input$gender)
    # Filter the dataset based on the selected distance
    run_data <- run_data %>% filter(Distance == input$distance)
    # Convert the user input from time to seconds
    user_time <- (input$hours * 3600) + (input$minutes * 60) + input$seconds
    
    # Extract percentile number (numeric value) from the Percentile column for later calculation
    run_data <- run_data %>% mutate(Percentile_Number = as.numeric(str_extract(Percentile, "\\d+")))
    # Calculate the user's percentile based on their input time
    your_percentile <- run_data %>% filter(Time_Seconds >= user_time) %>% slice(1) %>% pull(Percentile_Number)
    
    # If no percentile is found, assign 100th percentile (fastest)
    if (length(your_percentile) == 0) {
      your_percentile <- 100
    }
    
    # Calculate the percentage of runners slower than the user
    percentage <- 100 - your_percentile
    
    # Display the result: how much faster the user is compared to others
    output$result <- renderText({
      paste("You are faster than approximately", percentage, "% of runners in your category.")
    })
    
    # Display the distance and gender for the user's input
    output$text <- renderText(
      paste(input$gender, names(distance_labels)[match(as.numeric(input$distance), distance_labels)]," Finish Times")
    )
    
    # Display a summary of the user's finish time
    output$summary_text <- renderText(
      paste("Your Finish Time:", sprintf("%02d:%02d:%02d", input$hours, input$minutes, input$seconds))
    )
    
    # Display the table of percentiles and corresponding times
    output$table <- renderTable({
      run_data %>% select(Percentile, Time)
    })
  })
  
  # Event handler for the "Predict Time" button
  observeEvent(input$predict, {
    
    # Convert the shortest distance time to seconds
    shortest_time <- (input$shortest_hours * 3600) + (input$shortest_minutes * 60) + input$shortest_seconds
    
    # Use Riegel's formula to predict the time for the selected long distance
    predicted_time <- riegel_predict(shortest_time, as.numeric(input$shortest_distance), as.numeric(input$predicted_distance))
    predicted_time_formatted <- format(as.POSIXct(predicted_time, origin = "1970-01-01", tz = "UTC"), "%H:%M:%S")
    
    # Filter the dataset for the predicted distance and gender (all)
    run_data <- data_running %>% filter(Distance == input$predicted_distance)
    run_data <- run_data %>% filter(Gender == "All")
    
    # Display the predicted finish time for the selected long distance
    output$predicted_time <- renderText({
      paste("Predicted Finish Time for", names(distance_labels)[match(as.numeric(input$predicted_distance), distance_labels)], "is", predicted_time_formatted)
    })
    
    # Display the distance and gender for the user's input
    output$text_predict <- renderText(
      paste("All", names(distance_labels)[match(as.numeric(input$predicted_distance), distance_labels)]," Finish Times")
    )
    
    # Display the table of percentiles and corresponding times for the predicted distance
    output$table_predict <- renderTable({
      run_data %>% select(Percentile, Time)
    })
    
  })
}

# Launch of Shiny app with the UI and server defined above
shinyApp(ui, server)
