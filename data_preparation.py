import pandas as pd
import numpy as np

# Load the dataset
file_path = 'heart_disease.csv'
data = pd.read_csv(file_path)

# Print Data Schema
print(data.info())
print(data.describe())

# Check for missing values
missing_values = data.isnull().sum()
print(missing_values)

# Drop rows/columns with missing values
data = data.dropna()

# 3. Convert Data Types
# Convert 'CHDRisk' column from 'yes'/'no', convert it to 0 and 1
data['CHDRisk'] = data['CHDRisk'].map({'no': 0, 'yes': 1})

# Z-score normalization
def standardize_column(column):
    return (column - column.mean()) / column.std()

columns_to_standardize = ['totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
for column in columns_to_standardize:
    data[column] = standardize_column(data[column])

# Handle Outliers
# Example using IQR method
def remove_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    return column[(column >= (Q1 - 1.5 * IQR)) & (column <= (Q3 + 1.5 * IQR))]

for column in columns_to_standardize:
    data = data.loc[remove_outliers(data[column]).index]

# Creating age groups column
data['age_group'] = pd.cut(data['age'], bins=[0, 30, 40, 50, 60, 70, 80], labels=['0-30', '31-40', '41-50', '51-60', '61-70', '71-80'])

# Manually encode categorical variables
data['sex'] = data['sex'].map({'male': 0, 'female': 1})
data['smokingStatus'] = data['smokingStatus'].map({'no': 0, 'yes': 1})
data['diabetes'] = data['diabetes'].map({'no': 0, 'yes': 1})

# Removing Duplicates
data = data.drop_duplicates()

# Ensure all 'yes'/'no' columns are binary
binary_columns = ['BPMeds', 'prevalentStroke', 'prevalentHyp']
for col in binary_columns:
    data[col] = data[col].apply(lambda x: 1 if x else 0)

# Save cleaned data to a new CSV
cleaned_file_path = 'cleaned_heart_disease.csv'
data.to_csv(cleaned_file_path, index=False)
