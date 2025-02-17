import pandas as pd
import numpy as np
import math
from dateutil import parser
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# Specifying the file paths for Apple and Garmin datasets
file_path_apple = "C:/Users/fbeik/OneDrive/Desktop/Project/Activities_Apple_FB.csv"
file_path_garmin = "C:/Users/fbeik/OneDrive/Desktop/Project/Activities_Garmin_FB.csv"

# Loading the datasets from the specified file paths
data_apple = pd.read_csv(file_path_apple, sep=";")
data_garmin = pd.read_csv(file_path_garmin, sep=";")

# Exploratory Data Analysis: Printing the first few rows
print("Apple Data (first 5 rows):")
print(data_apple.head())
print("\nGarmin Data (first 5 rows):")
print(data_garmin.head())

# Renaming and transforming columns for the Apple dataset
data_apple = data_apple.rename(columns={
    "ï..ID": "ID",
    "Part Of Day": "PartOfDay",
    "Activity Type": "ActivityType",
    "Elapsed Time": "TimeSeconds",
    "Max Heart Rate": "MaxHeartRate",
    "Average Heart Rate": "AverageHeartRate",
    "Elevation Gain": "ElevationGain",
    "Elevation Loss": "ElevationLoss",
    "Relative Effort": "RelativeEffort"
})

# Ensuring the columns are numeric and round
data_apple['AverageHeartRate'] = pd.to_numeric(data_apple['AverageHeartRate'], errors='coerce')
data_apple['ElevationGain'] = pd.to_numeric(data_apple['ElevationGain'], errors='coerce')
data_apple['ElevationLoss'] = pd.to_numeric(data_apple['ElevationLoss'], errors='coerce')

data_apple['AverageHeartRate'] = data_apple['AverageHeartRate'].round(0)
data_apple['ElevationGain'] = data_apple['ElevationGain'].round(2)
data_apple['ElevationLoss'] = data_apple['ElevationLoss'].round(2)

# Renaming and transforming columns for the Garmin dataset
data_garmin = data_garmin.rename(columns={
    "ï..ID": "ID",
    "Time.of.day": "PartOfDay",
    "Type": "ActivityType",
    "Time": "TimeSeconds",
    "Max Heart Rate": "MaxHeartRate",
    "Avg. Heart Rate": "AverageHeartRate",
    "Elevation Gain": "ElevationGain",
    "Elevation Loss": "ElevationLoss",
    "Relative Effort": "RelativeEffort"
})

# Ensuring the columns are numeric and round
data_garmin['AverageHeartRate'] = pd.to_numeric(data_garmin['AverageHeartRate'], errors='coerce')
data_garmin['ElevationGain'] = pd.to_numeric(data_garmin['ElevationGain'], errors='coerce')
data_garmin['ElevationLoss'] = pd.to_numeric(data_garmin['ElevationLoss'], errors='coerce')

data_garmin['AverageHeartRate'] = data_garmin['AverageHeartRate'].round(0)
data_garmin['ElevationGain'] = data_garmin['ElevationGain'].round(2)
data_garmin['ElevationLoss'] = data_garmin['ElevationLoss'].round(2)


# Combining the Apple and Garmin datasets into a single dataset
data = pd.concat([data_apple, data_garmin], ignore_index=True)

# Exploratory Data Analysis: Printing the first few rows and a summary of the combined data
print("\nCombined Data (first 5 rows):")
print(data.head())
print("\nSummary of Combined Data:")
print(data.describe(include='all'))


# Checking for duplicates and get the total number of duplicates
duplicates_count = data.duplicated().sum()
print("\nTotal number of duplicate rows:", duplicates_count)

# Viewing the duplicated rows
duplicated_rows = data[data.duplicated()]
print("\nDuplicated rows:")
print(duplicated_rows)

# Removing duplicate rows and check how many were removed
duplicates_before = data.shape[0]
data = data.drop_duplicates()
duplicates_after = data.shape[0]
print("\nDuplicates removed:", duplicates_before - duplicates_after)

# removing rows where ID is NA
data = data.dropna(subset=['ID'])

# checking datatypes
print(data.info())

# Calculating pace per km and adding time in minutes
data['time_in_minutes'] = pd.to_numeric(data['TimeSeconds'], errors='coerce') / 60

# Replacing commas with periods in the 'Distance' column and convert to numeric
data['Distance'] = data['Distance'].astype(str).str.replace(',', '.')
data['Distance'] = pd.to_numeric(data['Distance'], errors='coerce')

# Calculating the pace (minutes per kilometer)
data['PacePerKM_value'] = data['time_in_minutes'] / data['Distance']

# Converting the calculated pace to a time format (MM:SS)
def convert_pace_to_mmss(pace):
    if pd.isna(pace) or pace <= 0 or np.isinf(pace):
        return None
    minutes = int(math.floor(pace))
    seconds = int(round((pace - minutes) * 60))
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes:02d}:{seconds:02d}"

data['PacePerKM'] = data['PacePerKM_value'].apply(convert_pace_to_mmss)

# ensuring the 'TimeSeconds' column is numeric and convert it to "HH:MM:SS" format
data['TimeSeconds'] = pd.to_numeric(data['TimeSeconds'], errors='coerce')
data = data.dropna(subset=['TimeSeconds'])
data['Time1'] = pd.to_datetime(data['TimeSeconds'], unit='s', origin='unix').dt.strftime('%H:%M:%S')

# Filtering for only running activities and exclude runs under 1 km
data_run = data[(data['ActivityType'] == "Run") & (data['Distance'] > 0.99)].copy()
data_run = data_run.sort_values(by='Distance', ascending=False)

def robust_date_parser(date_str):
    try:
        return pd.to_datetime(date_str, format='%b %d, %Y, %I:%M:%S %p')
    except Exception:
        pass 
    try:
        return parser.parse(date_str)
    except Exception:
        # If all parsing attempts fail, return a default date
        print(f"Warning: Could not parse date '{date_str}'. Using default date 1900-01-01.")
        return pd.Timestamp("1900-01-01")

# Applying the parser to create FormattedDate column
data_run['FormattedDate'] = data_run['Date'].apply(robust_date_parser)

# Optionally, convert to a specific string format
#data_run['FormattedDate'] = data_run['FormattedDate'].dt.strftime('%Y-%m-%d %H:%M:%S')

# checking again for null values
null_count = data_run['FormattedDate'].isnull().sum()
print("Number of null values in 'FormattedDate':", null_count)

# Creating the Weekday column from FormattedDate
data_run['Weekday'] = pd.to_datetime(data_run['FormattedDate'], format='%Y-%m-%d %H:%M:%S').dt.day_name()

# Checking the results
print(data_run[['Date', 'FormattedDate', 'Weekday']].head())


numeric_data = data_run.select_dtypes(include=[np.number])

# Display the first few rows
print(numeric_data.head())

# Selecting only the specified numeric columns
selected_columns = ["Calories", "MaxHeartRate", "AverageHeartRate", "RelativeEffort", "TimeSeconds", "Distance", "time_in_minutes", "PacePerKM_value"]
    
# Creating a new DataFrame with the selected columns
numeric_data_selected = data_run[selected_columns]

# Selecting numeric columns and compute the correlation matrix
cor_matrix = numeric_data_selected.corr()
print("Correlation Matrix:")
print(cor_matrix)

# Visualizing the correlation matrix using a heatmap - displaying only the lower triangle
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(cor_matrix, dtype=bool))  # mask for the upper triangle
sns.heatmap(cor_matrix, mask=mask, annot=True, cmap='coolwarm', square=True)
plt.title("Correlation Matrix (Lower Triangle)")
plt.show()

# analyzing running activity by weekday
weekday_summary = (
    data_run.groupby("Weekday")
    .agg(Runs=("Weekday", "size"), Avg_Distance=("Distance", "mean"))
    .reset_index()
    .sort_values(by="Runs", ascending=False)
)
print("Weekday Summary:")
print(weekday_summary)

# analyzing running activity by time of day
time_summary = (
    data_run.groupby("PartOfDay")
    .agg(Runs=("PartOfDay", "size"), Avg_Distance=("Distance", "mean"))
    .reset_index()
    .sort_values(by="Runs", ascending=False)
)
print("Time of Day Summary:")
print(time_summary)

# analyzing running activity by both weekday and time of day
all_summary = (
    data_run.groupby(["Weekday", "PartOfDay"])
    .agg(Runs=("ID", "size"), Avg_Distance=("Distance", "mean"))
    .reset_index()
)
# Sorting by Weekday and descending by Runs
all_summary = all_summary.sort_values(by=["Weekday", "Runs"], ascending=[True, False])
print("All Summary:")
print(all_summary)

# Converting 'PartOfDay' to a categorical type with a specified order
categories = ["Morning", "Lunch", "Afternoon", "Evening", "Night"]
all_summary["PartOfDay"] = pd.Categorical(all_summary["PartOfDay"], categories=categories, ordered=True)

# visualizing the number of runs by weekday and time of day
plt.figure(figsize=(10, 6))
sns.barplot(data=all_summary, x="Weekday", y="Runs", hue="PartOfDay", dodge=True)
plt.title("Läufe nach Wochentag und Tageszeit")
plt.xlabel("Wochentag")
plt.ylabel("Anzahl der Läufe")
plt.show()

# calculating average and total distance of all runs
average_distance = data_run['Distance'].mean()
total_distance = data_run['Distance'].sum()
print("Average Distance (km):", average_distance)
print("Total Distance (km):", total_distance)

# visualizing relationship between distance and time
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data_run, x="Distance", y="TimeSeconds")
sns.regplot(data=data_run, x="Distance", y="TimeSeconds", scatter=False, color="green")
plt.title("Relationship Between Distance and Time")
plt.show()

# Linear regression model
# Converting numerical columns with comma separators
for col in ["Distance", "MaxHeartRate", "AverageHeartRate", "ElevationGain", "Calories"]:
    data_run[col] = data_run[col].astype(str).str.replace(",", ".").astype(float)

print(data_run.isna().sum())  # Counting missing values in each column
# Filling missing values with mean of each column
data_run.fillna(data_run.mean(numeric_only=True), inplace=True)

# Selecting features and target variable
X = data_run[["Distance", "MaxHeartRate", "AverageHeartRate", "Calories"]]
y = data_run["TimeSeconds"]

# Train-Test Split (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Linear Regression Model
lm_model = LinearRegression()
lm_model.fit(X_train, y_train)

# Making predictions
y_pred = lm_model.predict(X_test)

# Evaluating model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Linear Regression RMSE:", rmse)

# On average, the model's predictions differ by ~7 min 56 sec from the actual running time
# model is making large errors (~7 min 56 sec) - therefore testing another model

# Training Random Forest Model
rf_model = RandomForestRegressor(n_estimators=500, random_state=42)
rf_model.fit(X_train, y_train)

# Making predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluating model
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print("Random Forest RMSE:", rmse_rf)

# Feature importance
feat_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_importances.sort_values().plot(kind="barh")
plt.title("Feature Importance in Random Forest")
plt.show()

# On average, the Random Forest model’s predictions are off by ~11 min 32 sec from the actual time

# XGBOOST
# Defining hyperparameter grid
param_grid = {
    "n_estimators": [100, 500, 1000],
    "max_depth": [3, 6, 10],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.7, 1.0]
}

# Grid Search for best parameters
xgb_model = xgb.XGBRegressor()
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring="neg_root_mean_squared_error")
grid_search.fit(X_train, y_train)

# Printing the best parameters
print("Best parameters:", grid_search.best_params_)

# Training model
xgb_best = xgb.XGBRegressor(**grid_search.best_params_)
xgb_best.fit(X_train, y_train)

# Predicting and calculating RMSE
y_pred_xgb_best = xgb_best.predict(X_test)
rmse_xgb_best = np.sqrt(mean_squared_error(y_test, y_pred_xgb_best))
print("Optimized XGBoost RMSE:", rmse_xgb_best)

# On average the XGBoost predictions are off by ~5 min 37 sec
# Outliers in Data – Large variation in running times can mislead the model
# Data might not be large enough – XGBoost works best with big datasets (thousands of rows).

# machine learning models struggle to generalize running performance
# Using Riegel’s formula - a well-established non-linear sports performance predictor
# Riegel's formula function
def riegel_predict(time_short, distance_short, distance_long, exponent=1.06):
    time_long = time_short * (distance_long / distance_short) ** exponent
    return time_long

# Filtering data for runs between 11km and 21km
filtered_runs = data_run[data_run["Distance"] > 11]

# Calculateing average running time for those runs
time_short_distance = filtered_runs["TimeSeconds"].mean()
short_distance = filtered_runs["Distance"].mean()

# Defining the target marathon distance
long_distance = 42.195  # Marathon distance in km

# Predicting marathon time
predicted_time_riegel = riegel_predict(time_short_distance, short_distance, long_distance)

# Converting seconds to HH:MM:SS format
predicted_marathon_time = pd.to_datetime(predicted_time_riegel, unit='s').strftime("%H:%M:%S")

print(f"Predicted Marathon Time: {predicted_marathon_time}")
