import pandas as pd

netflix = pd.read_csv("netflix1.csv")
print(netflix.head())
print(netflix.columns)
print(netflix["rating"])

# No missing values each column
print(sum([sum(netflix[column].isna()) for column in netflix.columns])) # Equals 0

# Examine Column Types of Dataset
column_types = {}

for column in netflix.columns:
    column_types[column] = str(netflix[column].dtype)
print(column_types)

'''

'''


