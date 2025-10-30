import pandas as pd
import pm4py
import os 
from pm4py.objects.conversion.log import converter as log_converter

#Reading the data from the xes file
xes_path = "/Users/omarhamdi/Desktop/BPIChallenge2017.xes"
log = pm4py.read_xes(xes_path)
df = pm4py.convert_to_dataframe(log)


# renaming the columns
df.rename(columns={
    'case:concept:name': 'CaseID',
    'concept:name': 'Activity',
    'time:timestamp': 'Timestamp'
}, inplace=True)


# Saving the data in a CSV file 
output_csv_path = "BPIChallenge2017.csv"
df.to_csv(output_csv_path, index=False)



#sort by CseID and then sort it by timestamp
df.sort_values(by=['CaseID', 'Timestamp'], inplace=True)
print("DataFrame sorted by CaseID and Timestamp.")


# Define all columns being that have missing values
imputation_cols = [
    'OfferedAmount', 'MonthlyCost', 'FirstWithdrawalAmount',
    'NumberOfTerms', 'CreditScore', 'Accepted', 'Selected', 'OfferID'
]
missing_before = df[imputation_cols].isnull().sum()
print(missing_before[missing_before > 0])

# Define columns identified as having high missing percentages
numeric_cols = [
    'OfferedAmount', 'MonthlyCost', 'FirstWithdrawalAmount',
    'NumberOfTerms', 'CreditScore'
]
categorical_cols = [
    'Accepted', 'Selected', 'OfferID'
]

# In numerical columns replace missing values with 0,
for col in numeric_cols:
    df[col].fillna(0.0, inplace=True)

# In categorical columns replace with NA
for col in categorical_cols:
    df[col].fillna('N/A', inplace=True)

print("\n--- Missing Value Count AFTER Imputation ---")
missing_after = df[imputation_cols].isnull().sum()
print(missing_after)


#printing the data
print(df.shape) 
print(df.head())
print(df.info())