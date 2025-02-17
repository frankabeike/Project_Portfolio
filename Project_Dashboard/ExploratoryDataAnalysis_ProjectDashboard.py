import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np

### Analysis America

# upload the data
data_america = pd.read_csv("your/file/path.csv")

# Exploratory Data Analysis
print(data_america.head())

# check for duplicates
data_america.duplicated().sum

# summary of data
print(data_america.dtypes)
data_america.describe()

# check for NA
data_america.isna().any()
data_america['Alcohol_Consumption'].isna().sum() # how many are NA
data_america[data_america['Alcohol_Consumption'].isna()] # which rows are NA in that column
data_america['Alcohol_Consumption'].unique() # which categories does Alcohol_Consumption has
data_america['Alcohol_Consumption'].fillna('None',inplace =True) # replace NA with the original value 'None'

# Assuming df is your DataFrame and it contains the columns 'Age_Group', 'Gender', and 'BMI_New'
def calculate_weight_status(row):
    # Female conditions
    if row['Age_Group'] == 'Youth' and row['Gender'] == 'Female':
        if row['BMI_New'] < 18:
            return "Underweight"
        elif row['BMI_New'] <= 24:
            return "Normal Weight"
        elif row['BMI_New'] <= 29:
            return "Overweight"
        elif row['BMI_New'] <= 39:
            return "Obesity"
        elif row['BMI_New'] > 39:
            return "Severe Obesity"
 
    elif row['Age_Group'] == 'Adult' and row['Gender'] == 'Female':
        if row['BMI_New'] < 20:
            return "Underweight"
        elif row['BMI_New'] <= 26:
            return "Normal Weight"
        elif row['BMI_New'] <= 31:
            return "Overweight"
        elif row['BMI_New'] <= 41:
            return "Obesity"
        elif row['BMI_New'] > 41:
            return "Severe Obesity"
   
    # Male conditions
    elif row['Age_Group'] == 'Adult' and row['Gender'] == 'Male':
        if row['BMI_New'] < 21:
            return "Underweight"
        elif row['BMI_New'] <= 27:
            return "Normal Weight"
        elif row['BMI_New'] <= 32:
            return "Overweight"
        elif row['BMI_New'] <= 42:
            return "Obesity"
        elif row['BMI_New'] > 42:
            return "Severe Obesity"
 
    elif row['Age_Group'] == 'Youth' and row['Gender'] == 'Male':
        if row['BMI_New'] < 19:
            return "Underweight"
        elif row['BMI_New'] <= 25:
            return "Normal Weight"
        elif row['BMI_New'] <= 30:
            return "Overweight"
        elif row['BMI_New'] <= 40:
            return "Obesity"
        elif row['BMI_New'] > 40:
            return "Severe Obesity"
   
    # Other Gender conditions
    elif row['Age_Group'] == 'Youth' and row['Gender'] == 'Other':
        if row['BMI_New'] < 18:
            return "Underweight"
        elif row['BMI_New'] <= 24:
            return "Normal Weight"
        elif row['BMI_New'] <= 29:
            return "Overweight"
        elif row['BMI_New'] <= 39:
            return "Obesity"
        elif row['BMI_New'] > 39:
            return "Severe Obesity"
 
    elif row['Age_Group'] == 'Adult' and row['Gender'] == 'Other':
        if row['BMI_New'] < 20:
            return "Underweight"
        elif row['BMI_New'] <= 26:
            return "Normal Weight"
        elif row['BMI_New'] <= 31:
            return "Overweight"
        elif row['BMI_New'] <= 41:
            return "Obesity"
        elif row['BMI_New'] > 41:
            return "Severe Obesity"
 
    # If no condition is met
    return "Unknown"
 
# Apply the function to your DataFrame
data_america['Weight_Status'] = data_america.apply(calculate_weight_status, axis=1)

# Convert categorical columns to numerical using label encoding
categorical_columns = ['Age_Group', 'Gender', 'Ethnicity', 'Smoking_Status', 'Alcohol_Consumption',
                       'Diet_Quality', 'Stress_Level', 'Air_Quality_Index', 'Income_Level','Weight_Status']
 
le = LabelEncoder()
for column in categorical_columns:
    data_america[column] = le.fit_transform(data_america[column])
 
# correlation matrix
corr_matrix = data_america.corr()
 
# Plotting the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()
 
sns.heatmap(data_america.corr().drop(['ID'], axis=1).drop(['ID'], axis=0), annot = True)
plt.show()
 
###################################################################################################################################
### Analysis Nigeria
 
# upload the data
data_nigeria = pd.read_csv("your/file/path.csv")
 
# Exploratory Data Analysis
print(data_nigeria.head())
 
# check for duplicates
data_nigeria.duplicated().sum
 
# summary of data
print(data_nigeria.dtypes)
data_nigeria.describe()
 
# Add ID column
data_nigeria.insert(0, 'ID',range(1,len(data_nigeria)+1))
 
# check for NA
data_nigeria.isna().any() # are there any NA
data_nigeria['Alcohol_Consumption'].isna().sum() # how many are NA
data_nigeria[data_nigeria['Alcohol_Consumption'].isna()] # which rows are NA in that column
data_nigeria['Alcohol_Consumption'].unique() # which categories does Alcohol_Consumption has
data_nigeria['Alcohol_Consumption'].fillna('None',inplace =True) # replace NA with the original value 'None'

