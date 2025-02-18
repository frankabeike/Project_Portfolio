import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np

# upload the data
data_america = pd.read_csv("your/file/path.csv")
data_nigeria = pd.read_csv("your/file/path.csv")
 

# defining plot functions
# function for histogram
def plot_histogram(data, column, bins=20, figsize=(8, 4), 
                   title=None, xlabel=None, ylabel="Frequency", color='skyblue'):
    
    plt.figure(figsize=figsize)
    sns.histplot(data[column], kde=False, bins=bins, color=color)
    
    if title is None:
        title = f'Histogram of {column}'
    if xlabel is None:
        xlabel = column
        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# function for density plot
def plot_density(data, column, figsize=(8, 4), 
                 title=None, xlabel=None, ylabel="Density", color='coral'):

    plt.figure(figsize=figsize)
    sns.kdeplot(data[column], shade=True, color=color)
    
    if title is None:
        title = f'Density Plot of {column}'
    if xlabel is None:
        xlabel = column
        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# function for boxplot
def plot_box(data, column, figsize=(8, 2), 
             title=None, xlabel=None, color='lightgreen'):

    plt.figure(figsize=figsize)
    sns.boxplot(x=data[column], color=color)
    
    if title is None:
        title = f'Box Plot of {column}'
    if xlabel is None:
        xlabel = column
        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.show()

def plot_count(data, column, figsize=(6, 4), 
               title=None, xlabel=None, ylabel="Count", palette='pastel'):
 
    plt.figure(figsize=figsize)
    sns.countplot(x=column, data=data, palette=palette)
    
    if title is None:
        title = f'Count Plot of {column}'
    if xlabel is None:
        xlabel = column
        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


### Analysis America
# Exploratory Data Analysis
print(data_america.head())

# check for duplicates
data_america.duplicated().sum

# summary of data
print(data_america.dtypes)
data_america.describe()
print(data_america.info())

# check for NA
data_america.isna().any()
data_america['Alcohol_Consumption'].isna().sum() # how many are NA
data_america[data_america['Alcohol_Consumption'].isna()] # which rows are NA in that column
data_america['Alcohol_Consumption'].unique() # which categories does Alcohol_Consumption has
data_america['Alcohol_Consumption'].fillna('None',inplace =True) # replace NA with the original value 'None'

# Assuming df is your DataFrame and it contains the columns 'Age_Group', 'Gender', and 'BMI'
def calculate_weight_status(row):
    # Female conditions
    if row['Age_Group'] == 'Youth' and row['Gender'] == 'Female':
        if row['BMI'] < 18:
            return "Underweight"
        elif row['BMI'] <= 24:
            return "Normal Weight"
        elif row['BMI'] <= 29:
            return "Overweight"
        elif row['BMI'] <= 39:
            return "Obesity"
        elif row['BMI'] > 39:
            return "Severe Obesity"
 
    elif row['Age_Group'] == 'Adult' and row['Gender'] == 'Female':
        if row['BMI'] < 20:
            return "Underweight"
        elif row['BMI'] <= 26:
            return "Normal Weight"
        elif row['BMI'] <= 31:
            return "Overweight"
        elif row['BMI'] <= 41:
            return "Obesity"
        elif row['BMI'] > 41:
            return "Severe Obesity"
   
    # Male conditions
    elif row['Age_Group'] == 'Adult' and row['Gender'] == 'Male':
        if row['BMI'] < 21:
            return "Underweight"
        elif row['BMI'] <= 27:
            return "Normal Weight"
        elif row['BMI'] <= 32:
            return "Overweight"
        elif row['BMI'] <= 42:
            return "Obesity"
        elif row['BMI'] > 42:
            return "Severe Obesity"
 
    elif row['Age_Group'] == 'Youth' and row['Gender'] == 'Male':
        if row['BMI'] < 19:
            return "Underweight"
        elif row['BMI'] <= 25:
            return "Normal Weight"
        elif row['BMI'] <= 30:
            return "Overweight"
        elif row['BMI'] <= 40:
            return "Obesity"
        elif row['BMI'] > 40:
            return "Severe Obesity"
   
    # Other Gender conditions
    elif row['Age_Group'] == 'Youth' and row['Gender'] == 'Other':
        if row['BMI'] < 18:
            return "Underweight"
        elif row['BMI'] <= 24:
            return "Normal Weight"
        elif row['BMI'] <= 29:
            return "Overweight"
        elif row['BMI'] <= 39:
            return "Obesity"
        elif row['BMI'] > 39:
            return "Severe Obesity"
 
    elif row['Age_Group'] == 'Adult' and row['Gender'] == 'Other':
        if row['BMI'] < 20:
            return "Underweight"
        elif row['BMI'] <= 26:
            return "Normal Weight"
        elif row['BMI'] <= 31:
            return "Overweight"
        elif row['BMI'] <= 41:
            return "Obesity"
        elif row['BMI'] > 41:
            return "Severe Obesity"
 
    # If no condition is met
    return "Unknown"
 
# Apply the function to your DataFrame
data_america['Weight_Status'] = data_america.apply(calculate_weight_status, axis=1)

# checking for distribution
# count plot for categorical columns - plot before changing string to numbers
plot_count(data_america, 'Weight_Status')
plot_count(data_america, 'Income_Level')
plot_count(data_america,'Alcohol_Consumption')
plot_count(data_america, 'Ethnicity')
plot_count(data_america, 'Diet_Quality')
plot_count(data_america,'Stress_Level')

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

# checking for outliers and distribution
# Plotting variable 'BMI' - just choosing dataset and variabel (column)
plot_histogram(data_america, 'BMI')
plot_density(data_america, 'BMI')
plot_box(data_america, 'BMI')

# Plotting variable 'Cholesterol_Level' - just choosing dataset and variabel (column)
plot_histogram(data_america, 'Cholesterol_Level')
plot_density(data_america, 'Cholesterol_Level')
plot_box(data_america, 'Cholesterol_Level')

# Plotting variable 'Heart Rate' - just choosing dataset and variabel (column)
plot_histogram(data_america, 'Heart_Rate')
plot_density(data_america, 'Heart_Rate')
plot_box(data_america, 'Heart_Rate')

 
###################################################################################################################################
### Analysis Nigeria
 

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

# Apply the function to your DataFrame
data_nigeria['Weight_Status'] = data_nigeria.apply(calculate_weight_status, axis=1)

# checking for distribution
# count plot for categorical columns - plot before changing string to numbers
plot_count(data_nigeria, 'Weight_Status')
plot_count(data_nigeria, 'Income_Level')
plot_count(data_nigeria,'Alcohol_Consumption')
plot_count(data_nigeria, 'Diet_Type')
plot_count(data_nigeria,'Stress_Level')
plot_count(data_nigeria,'Heart_Attack_Severity')
plot_count(data_nigeria,'Employment_Status')

# Convert categorical columns to numerical using label encoding
categorical_columns = ['Age_Group', 'Gender', 'Smoking_Status', 'Alcohol_Consumption','Familiy_History','Heart_Attack_Severity','Hospitalized','Survived',
                       'Diet_Type', 'Stress_Level', 'Income_Level','Weight_Status','Cholesterol_Level','Employment_Status','Exercise_Frequency','Hypertension','Diabetes']
 
le = LabelEncoder()
for column in categorical_columns:
    data_nigeria[column] = le.fit_transform(data_nigeria[column])
 
# correlation matrix
corr_matrix = data_nigeria.corr()
 
# Plotting the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# checking for outliers and distribution
# Plotting variable 'BMI' - just choosing dataset and variabel (column)
plot_histogram(data_nigeria, 'BMI')
plot_density(data_nigeria, 'BMI')
plot_box(data_nigeria, 'BMI')
