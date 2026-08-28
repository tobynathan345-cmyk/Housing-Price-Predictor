import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from paths import DATA_RAW

# Function to clean the raw data


def clean_raw_data(df):
    # Step 1: same as before, remove columns with more that ~40% missing data points.
    missing_counts = df.isnull().sum()
    columns_to_drop = missing_counts[missing_counts > 600].index
    df = df.drop(columns=columns_to_drop)

    # Step 2: Feature engineering, create a new variable TotalSF, SaleAge, TotalBath and LastRemod
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["SaleAge"] = df["YrSold"] - df["YearBuilt"]
    df["TotalBath"] = df["FullBath"] + 0.5*df["HalfBath"] + \
        df["BsmtFullBath"] + 0.5*df["BsmtHalfBath"]
    df["LastRemod"] = df["YrSold"] - df["YearRemodAdd"]

    df = df.drop(columns=["TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "YrSold",
                 "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath", "YearRemodAdd"])

    # Step 3: Manually assign the the year the house was built to the missing values in GarageYrBlt
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])

    df = df.drop(columns=["YearBuilt"])

    return df

# Function to build the preprocessing pipeline


def build_preprocessor(df):
    # Step 3.1: Build Imputers for the numerical and categorical columns with missing values and and values that don't apply
    median_transformer = SimpleImputer(strategy="median")
    numerical_missing_transformer = SimpleImputer(
        strategy="constant", fill_value=0)
    mode_transformer = SimpleImputer(strategy="most_frequent")
    categorical_missing_transformer = SimpleImputer(
        strategy="constant", fill_value="None")

    # Step 3.2: Build pipeline to impute and encode the categorical data so that it can be passed into our model
    none_fill_pipeline = Pipeline(steps=[
        ("impute", categorical_missing_transformer),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    mode_fill_pipeline = Pipeline(steps=[
        ("impute", mode_transformer),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Step 3.3: Organise the training data into the different categories that need to be imputed (and encoded for the categorical data)
    garage_bsmt_columns = [
        col for col in df.columns if "Garage" in col or "Bsmt" in col]
    garage_bsmt_numeric_columns = [
        col for col in garage_bsmt_columns
        if df[col].dtype in ("int64", "float64") and col != "GarageYrBlt"
    ]

    zero_fill_columns = garage_bsmt_numeric_columns + ["MasVnrArea"]

    categorical_none_columns = [
        col for col in df.columns
        if df[col].dtype == "str" and col != "Electrical"
    ]

    # Step 3.4: Our preprocessor which groups all the data prep tasks into one place to allow for easily reusable code
    preprocessor = ColumnTransformer(transformers=[
        ("median_fill", median_transformer, ["LotFrontage"]),
        ("mode_fill", mode_fill_pipeline, ["Electrical"]),
        ("zero_fill", numerical_missing_transformer, zero_fill_columns),
        ("none_fill", none_fill_pipeline, categorical_none_columns)
    ], remainder="passthrough")

    preprocessor.set_output(transform="pandas")

    return preprocessor
