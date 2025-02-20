import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency

# uploading the data
data_america = pd.read_csv("your/file/path.csv")
data_nigeria = pd.read_csv("your/file/path.csv")

#######################################################################################
# defining plot functions
# function for histogram
def plot_histogram(data, column, bins=20, figsize=(8, 4), 
                   title=None, xlabel=None, ylabel="Frequency", color='skyblue'):
    
    plt.figure(figsize=figsize)
    sns.histplot(data[column], kde=False, bins=bins, color=color)
    
    # naming plot & axis
    if title is None:
        title = f'Histogram of {column}'
    if xlabel is None:
        xlabel = column
    
    # Return plot     
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# function for density plot
def plot_density(data, column, figsize=(8, 4), 
                 title=None, xlabel=None, ylabel="Density", color='coral'):

    plt.figure(figsize=figsize)
    sns.kdeplot(data[column], shade=True, color=color)
    
    # naming plot & axis
    if title is None:
        title = f'Density Plot of {column}'
    if xlabel is None:
        xlabel = column
    
    # Return plot     
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# function for boxplot
def plot_box(data, column, figsize=(8, 2), 
             title=None, xlabel=None, color='lightgreen'):

    plt.figure(figsize=figsize)
    sns.boxplot(x=data[column], color=color)
    
    # naming plot & axis
    if title is None:
        title = f'Box Plot of {column}'
    if xlabel is None:
        xlabel = column
    
    # Return plot    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.show()

# function for categorical plot
def plot_count(data, column, figsize=(6, 4), 
               title=None, xlabel=None, ylabel="Count", palette='pastel'):
 
    plt.figure(figsize=figsize)
    sns.countplot(x=column, data=data, palette=palette)
    
    # naming plot & axis
    if title is None:
        title = f'Count Plot of {column}'
    if xlabel is None:
        xlabel = column
    
    # Return plot     
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# function for chi square test
def chi_square_test(data, col1, col2):

    # Create a contingency table
    contingency_table = pd.crosstab(data[col1], data[col2])

    # Perform Chi-Square Test
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

    # Interpretation
    significance = "Significant association" if p < 0.05 else "No significant association"

    # Print Results
    print(f"Chi-Square Test Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")
    print(f"Degrees of Freedom: {dof}")
    print(f"Interpretation: {significance}")

    # Return results as a dictionary
    return {
        "Chi-Square Statistic": chi2,
        "P-value": p,
        "Degrees of Freedom": dof,
        "Interpretation": significance
    }

# function for independet t test with plot
def independent_t_test(data1, data2, column_name, group1_name="Group 1", group2_name="Group 2"):

    # Extracting the numerical column values (removing NaN)
    group1_values = data1[column_name].dropna()
    group2_values = data2[column_name].dropna()

    # Calculating mean and standard deviation for each group
    mean1, mean2 = np.mean(group1_values), np.mean(group2_values)
    std1, std2 = np.std(group1_values, ddof=1), np.std(group2_values, ddof=1)
    n1, n2 = len(group1_values), len(group2_values)

    # Performing Independent T-Test
    t_stat, p_value = stats.ttest_ind(group1_values, group2_values)

    # Calculating mean difference
    mean_diff = mean1 - mean2

    # Computing 95% Confidence Interval
    se = np.sqrt((std1 ** 2 / n1) + (std2 ** 2 / n2))  # Standard error
    ci_low, ci_high = mean_diff - 1.96 * se, mean_diff + 1.96 * se  # 95% CI

    # Calculating Effect Size (Cohen’s d)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    cohen_d = mean_diff / pooled_std

    # Interpreting results
    significance = "Significant difference" if p_value < 0.05 else "No significant difference"

    # Print results
    print(f"T-Test Comparing {group1_name} vs. {group2_name}")
    print(f"T-Test Statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Mean Difference: {mean_diff:.4f}")
    print(f"95% Confidence Interval: ({ci_low:.4f}, {ci_high:.4f})")
    print(f"Effect Size (Cohen's d): {cohen_d:.4f}")
    print(f"Interpretation: {significance}\n")

    # Creating plots to visualize the data distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Boxplot
    axes[0].boxplot([group1_values, group2_values], labels=[group1_name, group2_name])
    axes[0].set_title(f"Boxplot of {column_name}")
    axes[0].set_ylabel(column_name)

    # Histogram
    min_val = min(np.min(group1_values), np.min(group2_values))
    max_val = max(np.max(group1_values), np.max(group2_values))
    bins = np.linspace(min_val, max_val, 20)
    axes[1].hist(group1_values, bins=bins, alpha=0.5, label=group1_name)
    axes[1].hist(group2_values, bins=bins, alpha=0.5, label=group2_name)
    axes[1].set_title(f"Histogram of {column_name}")
    axes[1].set_xlabel(column_name)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # Return results as a dictionary
    return {
        "T-Test Statistic": t_stat,
        "P-value": p_value,
        "Mean Difference": mean_diff,
        "Confidence Interval (95%)": (ci_low, ci_high),
        "Effect Size (Cohen's d)": cohen_d,
        "Interpretation": significance
    }

# function for chi square analysis
def chi_square_analysis(data, row_variable, col_variable):

    # Creating contingency table
    contingency_table = pd.crosstab(data[row_variable], data[col_variable])
    print("Contingency Table:")
    print(contingency_table)
    
    # Performing Chi-Square Test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-Square Test Statistic: {chi2:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Degrees of Freedom: {dof}")
    print("Expected Frequencies:")
    print(expected)
    
    # Calculating Cramér's V
    n = contingency_table.to_numpy().sum()  # Total number of observations
    min_dim = min(contingency_table.shape) - 1  # Minimum of (rows - 1, columns - 1)
    cramers_v = np.sqrt(chi2 / (n * min_dim))
    print(f"\nCramér's V: {cramers_v:.4f}")
    
    # Calculating Standardized Residuals
    observed = contingency_table.to_numpy()
    std_resid = (observed - expected) / np.sqrt(expected)
    std_resid_df = pd.DataFrame(std_resid, index=contingency_table.index, columns=contingency_table.columns)
    print("\nStandardized Residuals:")
    print(std_resid_df)
    
    # Return all results in a dictionary
    results = {
        "contingency_table": contingency_table,
        "chi2": chi2,
        "p_value": p_value,
        "degrees_of_freedom": dof,
        "expected": expected,
        "cramers_v": cramers_v,
        "standardized_residuals": std_resid_df
    }
    return results

#################################################################################################
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

 
# Creating a copy of original data for correlation analysis
data_america_corr = data_america.copy()
# Listing categorical columns 
categorical_columns = ['Age_Group', 'Gender', 'Ethnicity', 'Smoking_Status', 'Alcohol_Consumption',
                       'Diet_Quality', 'Stress_Level', 'Air_Quality_Index', 'Income_Level','Weight_Status']

# Initializing LabelEncoder
le = LabelEncoder()
for column in categorical_columns:
    data_america_corr[column] = le.fit_transform(data_america_corr[column]) # Encoding the categorical columns in the copy

# Creating the correlation matrix using copied dataframe
corr_matrix = data_america_corr.corr()
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

# statistical tests
# Chi square test gender & hear attack
result = chi_square_test(data_america, "Gender", "Heart_Attack") # no significant association
# Chi square test Air_Quality_Index & hear attack
result = chi_square_test(data_america, "Air_Quality_Index", "Heart_Attack") # no significant association
# Chi square test Alcohol consumption & hear attack
result = chi_square_test(data_america, "Alcohol_Consumption", "Heart_Attack") # no significant association
# Chi square test weight_status & hear attack
result = chi_square_test(data_america, "Weight_Status", "Heart_Attack") # no significant association
## With a different distribution (for example, if there were fewer people without a heart attack) the test could possibly show a significant difference if the deviations from the expected values were greater
results = chi_square_test(data_america, 'Heart_Attack', 'Family_History') # no significant association

# further analysis for association (no significant association can be due to the large no heart attack values)
results = chi_square_analysis(data_america, 'Heart_Attack', 'Weight_Status')
results = chi_square_analysis(data_america, 'Heart_Attack', 'Family_History')
 
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

# Creating a copy of original data for correlation analysis
data_nigeria_corr = data_nigeria.copy()
# Listing categorical columns 
categorical_columns = ['Age_Group', 'Gender', 'Smoking_Status', 'Alcohol_Consumption',
                       'Family_History', 'Heart_Attack_Severity', 'Hospitalized', 'Survived',
                       'State', 'Urban_Rural', 'Diet_Type', 'Stress_Level', 'Income_Level',
                       'Weight_Status', 'Cholesterol_Level', 'Employment_Status', 
                       'Exercise_Frequency', 'Hypertension', 'Diabetes']

# Initializing LabelEncoder
le = LabelEncoder()
for column in categorical_columns:
    data_nigeria_corr[column] = le.fit_transform(data_nigeria_corr[column]) # Encoding the categorical columns in the copy

# Creating the correlation matrix using copied dataframe
corr_matrix = data_nigeria_corr.corr()
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

# statistical tests
# Chi square test weight_status & hear attack severity
result = chi_square_test(data_nigeria, "Weight_Status", "Heart_Attack_Severity") # significant association
# Chi square test income level & hear attack severity
result = chi_square_test(data_nigeria, "Income_Level", "Heart_Attack_Severity") # no significant association
# Chi square test Employment status & hear attack severity
result = chi_square_test(data_nigeria, "Employment_Status", "Heart_Attack_Severity") # significant association
# Chi square test stress level & hear attack severity
result = chi_square_test(data_nigeria, "Stress_Level", "Heart_Attack_Severity") # no significant association

result = chi_square_test(data_nigeria, "Hypertension", "Heart_Attack_Severity") # no significant association

# further analysis for significant association
results = chi_square_analysis(data_nigeria, 'Heart_Attack_Severity', 'Employment_Status')
results = chi_square_analysis(data_nigeria, 'Heart_Attack_Severity', 'Weight_Status') # severe obesity shows that less mild cases and more moderate cases
# significant p-value but Cramers V close to zero, indicating that strength of association between weight status and heart attack severity is weak
results = chi_square_analysis(data_nigeria, 'Heart_Attack_Severity', 'Hypertension')

# independent t test of BMI
result = independent_t_test(data_america, data_nigeria, "BMI", "U.S.", "Nigeria")
# Effect size (Cohen's d) 0.2306 shows that the difference between the groups is small
# not much more directly comparable because of difference in categorical and numerical values
