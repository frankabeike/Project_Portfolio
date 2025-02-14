# import necessary libraries
import pandas as pd
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

# upload the data 
data = pd.read_csv("C:/Users/fbeik/OneDrive/Desktop/Project/cell_samples.csv")

# Exploratory Data Analysis
print(data.head())

# check for duplicates
data.duplicated().sum()

# summary of data / which values do the variables have
data.describe()

# variation

# Histograms
data.hist(figsize=(10, 8), bins=20, edgecolor="black")
plt.suptitle("Feature Distributions")
plt.show()

# Boxplots
plt.figure(figsize=(12, 6))
sns.boxplot(data=data.drop(columns=['Class']))
plt.xticks(rotation=45)
plt.title("Boxplot of Features")
plt.show()

# check the datatypes
print(data.dtypes) 
# check where BareNuc has no numeric values
special_rows = data[data['BareNuc'].isna() | (data['BareNuc'] == '?')]
print(special_rows['BareNuc'])
# 1. Ersetze '?' durch NaN
data['BareNuc'] = data['BareNuc'].replace('?', pd.NA)
# 2. Ersetze NaN-Werte mit 5
data['BareNuc'].fillna(5, inplace=True)
# 3. Konvertiere die 'BareNuc'-Spalte in einen numerischen Datentyp (z. B. int oder float)
data['BareNuc'] = pd.to_numeric(data['BareNuc'], errors='coerce')

# convert the columns into integers
for col in data.columns:
    pd.to_numeric(data[col], errors='coerce')
        
# Add ID column
data.insert(0, 'ID', range(1, len(data) + 1))

# Check if "Class" column has any missing values
data['Class'].isna().any()

# Replace NA values in all columns except "Class" with 5 (there are no Null values)
data.loc[:, data.columns != 'Class'] = data.loc[:, data.columns != 'Class'].fillna(5)

# check the balance / imbalance of the classes / class distribution
total_rows = len(data)
benign = (data['Class'] == 2).sum()
malignant = (data['Class'] == 4).sum()
# visualization
data_vis = data
if 'ID' in data_vis.columns:
    data_vis.drop(columns=['ID'], inplace=True)
sns.countplot(x='Class', data=data_vis)
plt.title("Class Distribution")
plt.show()

benign_percent = (benign / total_rows) * 100
malignant_percent = (malignant / total_rows) * 100

# correlation matrix
corr = data.corr()
# Create a Mask for the Upper Triangle
mask = np.triu(np.ones_like(corr, dtype=bool))
# Set Up the Figure
plt.figure(figsize=(8, 6))
# Draw the Heatmap with the Mask
sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


# create balanced classes by oversampling class 4
# Features (X) and Labels (y) for Class 4
X_class = data.drop(columns='Class')
y_class = data['Class']
# Apply SMOTE only to Class 4 to generate synthetic samples
smote = SMOTE(sampling_strategy='auto', random_state=42)  # 'auto' means balance all classes to the minority
X_resampled, y_resampled = smote.fit_resample(X_class, y_class)
# Convert resampled X back to DataFrame
balanced_data = pd.DataFrame(X_resampled, columns=X_class.columns)
balanced_data['Class'] = y_resampled
# Shuffle the dataset 
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# check again for the balance / imbalance of the classes
bal_total_rows = len(balanced_data)
bal_benign = (balanced_data['Class'] == 2).sum()
bal_malignant = (balanced_data['Class'] == 4).sum()

bal_benign_percent = (bal_benign / bal_total_rows) * 100
bal_malignant_percent = (bal_malignant / bal_total_rows) * 100

# assign the data for a dataset used in training and testing
dataset = balanced_data

# Erstelle eine Funktion zur Anzeige der Confusion Matrix
def plot_conf_matrix(y_test, y_pred, title, ax):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"], cbar = False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    ax.xaxis.set_label_position('top') # Move x-axis label to top
    ax.xaxis.tick_top()  # Move x ticks to top

def results_model(y_test, y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


X = dataset.drop(columns=['Class'])  # Drop the target column
y = dataset['Class']                 # Target variable (2 = benign, 4 = malignant)


# decision tree
# split data into training and testing sets
# Partitioning: 70 / 30 -> test whats best partitioning
# 80 /20  accuracy & cohen‘s kappa sank (LR: accuracy same but cohen‘s kapp 0,001 lower
# 75 / 25  only accuracy and cohen‘s kappa of logistic regression was higher than 70 / 30
X_train, X_test, y_dt_train, y_dt_test = train_test_split(X, y, test_size=0.3, random_state=42)

# training decision tree model
# Uses Gini impurity for decision-making
# max_depth = 5 limits tree depth to prevent overfitting
# random_state = 42 ensures reproducibility
dt_clf = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42) 
dt_clf.fit(X_train, y_dt_train)

# make predicitons
y_pred_dt = dt_clf.predict(X_test)

# evaluate model performance
results_model(y_dt_test, y_pred_dt)

accuracy_dt = accuracy_score(y_dt_test, y_pred_dt)
report_dt = classification_report(y_dt_test, y_pred_dt, output_dict=True)
metrics_dt = extract_classification_metrics(report_dt)
metrics_dt["Model"] = "Decision Tree"
metrics_dt["Accuracy"] = accuracy_dt
results.append(metrics_dt)

# visualize the decision tree
plt.figure(figsize=(12, 6))
plot_tree(clf, feature_names=X.columns, class_names=["Benign (2)", "Malignant (4)"], filled=True)
plt.show()

# confusion matrix visualization
fig, axes = plt.subplots(1, 1, figsize=(5, 4))  # 1 Zeile, 1 Spalte
plot_conf_matrix(y_dt_test, y_pred_dt, "Decision Tree", axes)
plt.show()


# random forest
X_train, X_test, y_rf_train, y_rf_test = train_test_split(X, y, test_size=0.3, random_state=42)

# training random forest
# n_estimators = 100 number of decision trees in the forest
# Uses Gini impurity for split selection.
# max_depth = 5 limits tree depth to prevent overfitting
# random_state = 42 Ensures reproducibility
rf_clf = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=5, random_state=42)
rf_clf.fit(X_train, y_rf_train)

# make predictions
y_pred_rf = rf_clf.predict(X_test)

# evaluate model performance
results_model(y_rf_test, y_pred_rf)

accuracy_rf = accuracy_score(y_rf_test, y_pred_rf)
report_rf = classification_report(y_rf_test, y_pred_rf, output_dict=True)
metrics_rf = extract_classification_metrics(report_rf)
metrics_rf["Model"] = "Random Forest"
metrics_rf["Accuracy"] = accuracy_rf
results.append(metrics_rf)

# confusion matrix visualization
fig, axes = plt.subplots(1, 1, figsize=(5, 4))  # 1 Zeile, 1 Spalte
plot_conf_matrix(y_rf_test, y_pred_rf, "Random Forest", axes)
plt.show()

# Feature importance
importances = rf_clf.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10,5))
sns.barplot(x=importances, y=feature_names)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in Random Forest")
plt.show()


# training logistic regression
# Convert Target Variable to Binary (0 & 1)
y_lr = y.map({2: 0, 4: 1})

# split data into Training and Testing Sets using the binary classes (0 & 1)
X_train, X_test, y_lr_train, y_lr_test = train_test_split(X, y_lr, test_size=0.3, random_state=42)

# max_iter=1000 → Ensures sufficient iterations for convergence.
# solver='liblinear' → Works well for small datasets.
# random_state=42 → Ensures reproducibility.
log_reg = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
log_reg.fit(X_train, y_lr_train)

# make predictions
y_pred_log = log_reg.predict(X_test)

# evaluate model performance
results_model(y_lr_test, y_pred_log)

accuracy_lr = accuracy_score(y_lr_test, y_pred_log)
report_lr = classification_report(y_lr_test, y_pred_log, output_dict=True)
metrics_lr = extract_classification_metrics(report_lr)
metrics_lr["Model"] = "Logistic Regression"
metrics_lr["Accuracy"] = accuracy_lr
results.append(metrics_lr)

# confusion matrix visualization
fig, axes = plt.subplots(1, 1, figsize=(5, 4))  # 1 Zeile, 1 Spalte
plot_conf_matrix(y_lr_test, y_pred_log, "Logistic Regression", axes)
plt.show()

# feature importance (coefficients) --> muss ID gedroppt werden / kann ID gedroppt werden?
coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": log_reg.coef_[0]})
coefficients = coefficients.sort_values(by="Coefficient", ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(x="Coefficient", y="Feature", data=coefficients)
plt.title("Feature Importance in Logistic Regression")
plt.show()

# confusion matrix of all 3 models next to each other
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 Zeile, 3 Spalten
plot_conf_matrix(y_dt_test, y_pred_dt, "Decision Tree", axes[0])
plot_conf_matrix(y_rf_test, y_pred_rf, "Random Forest", axes[1])
plot_conf_matrix(y_lr_test, y_pred_log, "Logistic Regression", axes[2])
plt.tight_layout()
plt.show()

# put the accuracy and classification report in one table for interpretation which model is the most suitable

pd.DataFrame(results)

