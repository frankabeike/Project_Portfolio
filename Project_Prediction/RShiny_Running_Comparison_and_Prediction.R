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
# Function to calculate pace per km (mm:ss) given time in seconds and distance in km
calculate_pace <- function(Time_Seconds, Distance) {
  time_minutes <- Time_Seconds / 60 # convert seconds to minutes
  pace_value <- time_minutes / Distance # Calculate the pace (minutes per kilometer)
  pace_minutes <- floor(pace_value) # Extract whole minutes
  pace_seconds <- round((pace_value - pace_minutes)*60) # Calculate remaining seconds
  # Adjust in case rounding makes seconds = 60
  if (pace_seconds == 60) {
    pace_minutes <- pace_minutes + 1
    pace_seconds <- 0
  }
  sprintf("%02d:%02d",pace_minutes,pace_seconds) # Format as MM:SS
}


# Define UI for Shiny application
ui <- navbarPage(
  title = "Running Performance App",  # Navbar title
  theme = shinytheme("sandstone"),       # Apply a theme
  
  # First Tab - About
  tabPanel("About",
           fluidPage(
             h3("Running Performance App"),
             p("This app helps runners analyze their percentile ranking and predict race times using Riegel's formula."),
             p("Select your distance, enter your time, and find out how you compare to others."),
             p("Enjoy!")
           )
  ),
  
  # Second Tab - Percentile Calculator
  tabPanel("Percentile Calculator", # Title of the application
    
    # Sidebar layout with input controls for selecting distance, gender, and inputting time
    sidebarLayout(
      sidebarPanel(
        selectInput("distance", "Select Distance", choices = distance_labels), # Dropdown menu for selecting running distance
        bsTooltip("distance", "Choose a race distance.", "right"),
        selectInput("gender", "Select Gender", choices = c("All", "Female", "Male")), # Dropdown menu for selecting gender
        bsTooltip("gender", "Select your gender to compare against similar runners.", "right"),
        numericInput("hours", "Hours", value = 0, min = 0), # Numeric input for entering hours
        bsTooltip("hours", "Input the hours of your finish time for the selected race distance.", "right"),
        numericInput("minutes", "Minutes", value = 0, min = 0, max = 59), # Numeric input for entering minutes
        bsTooltip("minutes", "Input the minutes of your finish time for the selected race distance.", "right"),
        numericInput("seconds", "Seconds", value = 0, min = 0, max = 59), # Numeric input for entering seconds for the time
        bsTooltip("seconds", "Input the seconds of your finish time for the selected race distance.", "right"),
        actionButton("calculate", "Calculate Percentile") # Action button to calculate the percentile
      ),
      # Main panel for displaying results and outputs
      mainPanel(
        verbatimTextOutput("result"), # Display percentile result
        verbatimTextOutput("summary_text"), # Display Summary of Finish Time 
        verbatimTextOutput("pace"), # Display pace per Km
        textOutput("text"), # Display title of table
        tableOutput("table") # Display table with percentile and times
      ))
    ),
    # Third Tab - Time Prediction using Riegel's formula
  tabPanel("Time Prediction",
    sidebarLayout(
      sidebarPanel(
        numericInput("shortest_distance", "Enter Shortest Distance (km)", value = 5, min = 1), # Numeric inputs for entering shortest distance
        bsTooltip("shortest_distance", "Enter the known shorter race distance.", "right"),
        numericInput("shortest_hours", "Shortest Distance Hours", value = 0, min = 0), # Numeric inputs for entering hours
        bsTooltip("shortest_hours", "Enter the hours of your finish time for the shorter race.", "right"),
        numericInput("shortest_minutes", "Shortest Distance Minutes", value = 0, min = 0, max = 59), # Numeric inputs for entering minutes
        bsTooltip("shortest_minutes", "Enter the minutes of your finish time for the shorter race.", "right"),
        numericInput("shortest_seconds", "Shortest Distance Seconds", value = 0, min = 0, max = 59), # Numeric inputs for entering seconds
        bsTooltip("shortest_seconds", "Enter the seconds of your finish time for the shorter race.", "right"),
        selectInput("predicted_distance", "Select Predicted Distance", choices = distance_labels), # Dropdown menu for selecting the predicted distance
        bsTooltip("predicted_distance", "Choose a longer race distance for which you want to estimate your finish time.", "right"),
        actionButton("predict", "Predict Time") # Action button to predict time based on Riegel's formula
      ),
      # Main panel for displaying predicted time and percentile table
      mainPanel(
        verbatimTextOutput("predicted_time"), # Display predicted time
        verbatimTextOutput("predicted_pace"), # Display of average pace needed to reach predicted time
        textOutput("text_predict"), # Display title of table
        tableOutput("table_predict") # Display table with percentile and times
    )
  
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
    
    # calculating user's pace per km using the function
    user_pace <- calculate_pace(user_time,as.numeric(input$distance))
    
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
    output$text <- renderText({
      paste(input$gender, names(distance_labels)[match(as.numeric(input$distance), distance_labels)]," Finish Times")
    })
    
    # Display a summary of the user's finish time
    output$summary_text <- renderText({
      paste("Your Finish Time:", sprintf("%02d:%02d:%02d", input$hours, input$minutes, input$seconds))
    })
    
    output$pace <- renderText({
      paste("Your Pace per KM:",user_pace,"(MM:SS per KM)")
    })
    
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
    
    # calculating predicted pace per km using the helper function
    predicted_pace <- calculate_pace(predicted_time,as.numeric(input$predicted_distance))
    
    # Filter the dataset for the predicted distance and gender (all)
    run_data <- data_running %>% filter(Distance == input$predicted_distance)
    run_data <- run_data %>% filter(Gender == "All")
    
    # Display the predicted finish time for the selected long distance
    output$predicted_time <- renderText({
      paste("Predicted Finish Time for", names(distance_labels)[match(as.numeric(input$predicted_distance), distance_labels)], "is", predicted_time_formatted)
    })
    
    output$predicted_pace <- renderText({
      paste("Required pace per KM:",predicted_pace,"(MM:SS per KM)")
    })
    
    # Display the distance and gender for the user's input
    output$text_predict <- renderText({
      paste("All", names(distance_labels)[match(as.numeric(input$predicted_distance), distance_labels)]," Finish Times")
    })
    
    # Display the table of percentiles and corresponding times for the predicted distance
    output$table_predict <- renderTable({
      run_data %>% select(Percentile, Time)
    })
    
  })
}

# Launch of Shiny app with the UI and server defined above
shinyApp(ui, server)
