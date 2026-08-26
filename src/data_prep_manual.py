import os
import pandas as pd
from paths import DATA_RAW

train_df = pd.read_csv(os.path.join(DATA_RAW, "train.csv"))

# Remove columns with more that ~40% missing data points.
missing_counts = train_df.isnull().sum()
columns_to_drop = missing_counts[missing_counts > 600].index
train_df = train_df.drop(columns=columns_to_drop)

# Remove the massive outlier in row 1298
train_df = train_df.drop(index=1298)

# Making lists for the numerical and categorical columns
numeric_columns = [
    col for col in train_df.columns if train_df[col].dtype in ("int64", "float64")]
categorical_columns = list(set(train_df.columns) - set((numeric_columns)))

# Filling the missing values in the categorical columns
missing_categorical = train_df[categorical_columns].isnull().sum()
categorical_with_missing = missing_categorical[missing_categorical > 0].index

# Electricity is the only categorical column with actual missing values. Therefore we will fill these missing values with the mode
train_df["Electrical"] = train_df["Electrical"].fillna(
    train_df["Electrical"].mode()[0])

# The rest of the categorical columns either have no basement or no garage so we fill the missing values with None
categorical_none_columns = [
    col for col in categorical_with_missing if col != "Electrical"]
for col in categorical_none_columns:
    train_df[col] = train_df[col].fillna("None")

# Now we must fill all of the missing values in the numerical columns
garage_columns = [col for col in train_df.columns if "Garage" in col]
garage_missing_numerical = [
    col for col in garage_columns
    if train_df[col].dtype in ("int64", "float64") and train_df[col].isnull().sum() > 0 and col != "GarageYrBlt"
]

for col in garage_missing_numerical:
    train_df[col] = train_df[col].fillna(0)

train_df["GarageYrBlt"] = train_df["GarageYrBlt"].fillna(train_df["YearBuilt"])

basement_columns = [col for col in train_df.columns if "Bsmt" in col]

basement_numeric_missing = [
    col for col in basement_columns
    if train_df[col].dtype in ("int64", "float64") and train_df[col].isnull().sum() > 0
]
for col in basement_numeric_missing:
    train_df[col] = train_df[col].fillna(0)

train_df["MasVnrArea"] = train_df["MasVnrArea"].fillna(0)

train_df["LotFrontage"] = train_df["LotFrontage"].fillna(
    train_df["LotFrontage"].median())

missing = train_df.isnull().sum()
print(missing[missing > 0])
