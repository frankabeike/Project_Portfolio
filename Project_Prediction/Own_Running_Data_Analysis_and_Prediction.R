
install.packages('pacman')

library(pacman)
pacman:: p_load(ggplot2,scales,dplyr,DT,sqldf,readxl,corrplot)

# Specify the file paths for the Apple and Garmin datasets
file_path_apple <- "C:/Users/fbeik/OneDrive/Desktop/Project/Activities_Apple.csv"
file_path_garmin <- "C:/Users/fbeik/OneDrive/Desktop/Project/Activities_Garmin.csv"

# Load the datasets from the specified file paths
data_apple <- read.csv(file_path_apple,  sep = ";", fill=TRUE)
data_garmin <- read.csv(file_path_garmin,  sep = ";", fill=TRUE)


# Exploratory Data Analysis
print(head(data_apple))
print(head(data_garmin))

data_apple <- sqldf (" SELECT `ï..ID` AS ID, Date, `Part.Of.Day` AS PartOfDay, `Activity.Type` AS ActivityType, Distance, `Elapsed.Time` AS TimeSeconds, `Max.Heart.Rate` AS MaxHeartRate, round(`Average.Heart.Rate`) AS AverageHeartRate, Calories,round(`Elevation.Gain`,2) AS ElevationGain, round(`Elevation.Loss`,2) AS ElevationLoss, `Relative.Effort` AS RelativeEffort FROM data_apple")
data_garmin <- sqldf (" SELECT `ï..ID` AS ID, Date, `Time.of.day` AS PartOfDay, `Type` AS ActivityType, Distance, `Time` AS TimeSeconds, `Max.Heart.Rate` AS MaxHeartRate, round(`Avg..Heart.Rate`) AS AverageHeartRate, Calories,round(`Elevation.Gain`,2) AS ElevationGain, round(`Elevation.Loss`,2) AS ElevationLoss, `Relative.Effort` AS RelativeEffort FROM data_garmin")

# Combine the Apple and Garmin datasets into a single dataset
data <- sqldf("
    SELECT * 
    FROM data_apple
    UNION ALL
    SELECT *
    FROM data_garmin
")

# Exploratory Data Analysis
print(head(data))
summary(data)

# Check for duplicates and get the total number of duplicates
sum(duplicated(data))
# View the duplicated rows
data[duplicated(data), ]
# removing duplicate data aand check how many duplicates were removed
duplicates_before <- nrow(data)
data <- unique(data)
duplicates_after <- nrow(data)
cat("Duplicates removed:", duplicates_before - duplicates_after, "\n")


# Remove rows where the 'ID' column is NA
data <- data %>% filter(!is.na(ID))

# check the datatypes
str(data)

### Calculation for Pacer per KM
# Add a new column to convert time from seconds to minutes
data$time_in_minutes <- data$TimeSeconds / 60
# Replace commas with periods in the 'Distance' column and make it numeric
data$Distance <- as.numeric(gsub(",", ".", data$Distance))
# Calculate the pace (minutes per kilometer)
data$PacePerKM <- data$time_in_minutes / as.numeric(data$Distance)
# Convert the calculated pace to a time format (MM:SS)
data$PacePerKM <- sapply(data$PacePerKM, function(pace) {
  minutes <- as.integer(floor(pace))  # Extract whole minutes
  seconds <- as.integer(round((pace - minutes) * 60))  # Calculate remaining seconds
  sprintf("%02d:%02d", minutes, seconds)  # Format as MM:SS
})


# # Ensure the 'Time' column is numeric and convert it to "HH:MM:SS" format
data$TimeSeconds <- as.numeric(data$TimeSeconds)
data <- data[!is.na(data$TimeSeconds), ]  # Remove rows with NA
data$Time1 <- format(as.POSIXct(data$TimeSeconds, origin = "1970-01-01", tz = "UTC"), "%H:%M:%S")

# Filter for only running activities and exclude runs under 1 km; sort by distance
data_run <- sqldf("SELECT ID, Date, PartOfDay, ActivityType, Distance, TimeSeconds,`Time1` AS Time, PacePerKM, MaxHeartRate, AverageHeartRate, Calories, ElevationGain, ElevationLoss, RelativeEffort  FROM data")

data_run <- data %>%
  filter(ActivityType == "Run", Distance > 0.99) %>%
  arrange(desc(Distance))

# Convert the 'Date' column to a new format and add the weekday information
Sys.setlocale("LC_TIME", "C")
data_run$FormattedDate <- format(strptime(data_run$Date, format = "%b %d, %Y, %I:%M:%S %p"),"%Y-%m-%d %H:%M:%S")
WeekdayDate <- format(strptime(data_run$FormattedDate, format = "%Y-%m-%d %H:%M:%S"), "%m/%d/%Y")
data_run$Weekday <- weekdays(WeekdayDate)

# Correlation matrix to understand relationships between numerical features
cor_matrix <- cor(data_run[, sapply(data_run, is.numeric)])
print(cor_matrix)
# Visualize correlation matrix
corrplot(cor_matrix, method = "circle",type = "lower")


# Analyze running activity by weekday: count runs and calculate the average distance
weekday_summary <- data_run %>%
  group_by(Weekday) %>%
  summarise(Runs = n(),
            Avg_Distance = mean(Distance)) %>%
  arrange(desc(Runs))

print(weekday_summary)

# Analyze running activity by time of day: count runs and calculate the average distance
time_summary <- data_run %>%
  group_by(PartOfDay) %>%
  summarise(
    Runs = n(),
    Avg_Distance = mean(Distance)) %>%
  arrange(desc(Runs))

print(time_summary)

# Analyze running activity by both weekday and time of day
all_summary <- data_run %>%
  group_by(Weekday, PartOfDay) %>%
  summarise(
    Runs = n(),
    Avg_Distance = mean(Distance)) %>%
  arrange(Weekday,desc(Runs))

print(all_summary, n=27)

# Convert 'PartOfDay' into a factor with a specific order for visualization
all_summary$PartOfDay <- factor(all_summary$PartOfDay, levels = c("Morning", "Lunch", "Afternoon", "Evening", "Night"),exclude = NULL)

# Visualize the number of runs by weekday and time of day
ggplot(data = all_summary, aes(x = Weekday, y = Runs, fill = PartOfDay)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Läufe nach Wochentag und Tageszeit", y = "Anzahl der Läufe", x = "Wochentag") +
  theme_minimal()



# Count runs with and without elevation change, indicating whether they were indoors or outdoors
no_elevation_change <- nrow(data_run[data_run$ElevationGain == 0.0 & data_run$ElevationLoss == 0.0, ])
with_elevation_change <- nrow(data_run[data_run$ElevationGain > 0 | data_run$ElevationLoss > 0, ])
# Add a column to the table to show if the run was inside or outside
data_run$RunCondition <- ifelse(data_run$ElevationGain == 0 & data_run$ElevationLoss == 0, 
  "inside", 
  "outside"
)

cat("Runs without elevation change (inside):", no_elevation_change, "\n")
cat("Runs with elevation change (outside):", with_elevation_change, "\n")

# Calculate percentage of inside and outside runs
total_runs <- nrow(data_run)
percentage_inside <- (no_elevation_change / total_runs) * 100
percentage_outside <- (with_elevation_change / total_runs) * 100

cat("Percentage of runs inside:", round(percentage_inside, 2), "%\n")
cat("Percentage of runs outside:", round(percentage_outside, 2), "%\n")



# Calculate the average distance and the total distance of all runs
average_distance <- mean(data_run$Distance)
total_distance  <- sum(data_run$Distance)

# Add split times data from another file and join it with the running data
file_path_splits <- "C:/Users/fbeik/OneDrive/Desktop/Project/Activities_splits.csv"
data_splits <- read.csv(file_path_splits,  sep = ";", fill=TRUE)
data_splits <- sqldf("SELECT `ï..ID` AS ID, Split1, Split2, Split3, Split4, Split5, Split6, Split7, Split8, Split9, Split10, Split11, Split12, Split13, Split14, Split15, Split16, Split17, Split18, Split19, Split20, Split21, Split22 FROM data_splits")
joined_data <- left_join(data_run, data_splits, by = "ID")

joined_data$Distance <- as.numeric(joined_data$Distance)
# Visualize the relationship between distance and time to determine if it is linear or non-linear
ggplot(joined_data, aes(x = Distance, y = Time)) +
  geom_point() +
  geom_smooth(method = "lm", color = "green") +
  labs(title = "Relationship Between Distance and Time")

# Predict marathon time using Riegel's formula (non-linear extrapolation)
riegel_predict <- function(TimeShortDistance, ShortDistance, LongDistance, exponent = 1.06) {
  TimeLongDistance <- TimeShortDistance * (LongDistance / ShortDistance)^exponent
  return(TimeLongDistance)
}


# Predict marathon time based on average runs between 11 and 21 km
TimeShortDistance <- mean(joined_data$TimeSeconds[joined_data$Distance > 11]) 
ShortDistance <- mean(joined_data$Distance[joined_data$Distance > 11])
LongDistance <- 42.195
predicted_time_riegel <- riegel_predict(TimeShortDistance, ShortDistance, LongDistance)
Predicted_Marathon_Time <- format(as.POSIXct(predicted_time_riegel, origin = "1970-01-01", tz = "UTC"), "%H:%M:%S")
cat("Predicted Marathon Time (seconds):", Predicted_Marathon_Time)
