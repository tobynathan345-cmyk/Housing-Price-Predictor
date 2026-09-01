import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from paths import DATA_RAW

# Function to clean the raw data


def clean_raw_data(df):
    # same as before, remove columns with more that ~40% missing data points.
    columns_to_drop = ["PoolQC", "MiscFeature",
                       "Alley", "Fence", "MasVnrType", "FireplaceQu"]
    df = df.drop(columns=columns_to_drop, errors="ignore")

    # Feature engineering, create a new variable TotalSF, SaleAge, TotalBath and LastRemod
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["SaleAge"] = df["YrSold"] - df["YearBuilt"]
    df["TotalBath"] = df["FullBath"] + 0.5*df["HalfBath"] + \
        df["BsmtFullBath"] + 0.5*df["BsmtHalfBath"]
    df["LastRemod"] = df["YrSold"] - df["YearRemodAdd"]

    df = df.drop(columns=["TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "YrSold",
                 "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath", "YearRemodAdd"])

    # Manually assign the the year the house was built to the missing values in GarageYrBlt
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])

    df = df.drop(columns=["YearBuilt"])

    return df

# Function to build the preprocessing pipeline


def build_preprocessor(df):
    # Build Imputers for the numerical and categorical columns with missing values and and values that don't apply
    median_transformer = SimpleImputer(strategy="median")
    numerical_missing_transformer = SimpleImputer(
        strategy="constant", fill_value=0)
    mode_transformer = SimpleImputer(strategy="most_frequent")
    categorical_missing_transformer = SimpleImputer(
        strategy="constant", fill_value="None")

    # Build pipeline to impute and encode the categorical data so that it can be passed into our model
    none_fill_pipeline = Pipeline(steps=[
        ("impute", categorical_missing_transformer),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    mode_fill_pipeline = Pipeline(steps=[
        ("impute", mode_transformer),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Organise the training data into the different categories that need to be imputed (and encoded for the categorical data)
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

    all_handled_columns = (["LotFrontage", "Electrical"] +
                           zero_fill_columns + categorical_none_columns)

    remaining_columns = [
        col for col in df.columns if col not in all_handled_columns]

    remaining_numeric = [
        col for col in remaining_columns if df[col].dtype in ("int64", "float64")]

    remaining_categorical = [
        col for col in remaining_columns if df[col].dtype == "str"]

    catch_all_numeric = SimpleImputer(strategy="median")
    catch_all_categorical_pipeline = Pipeline(steps=[
        ("impute", mode_transformer),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Our preprocessor which groups all the data prep tasks into one place to allow for easily reusable code
    preprocessor = ColumnTransformer(transformers=[
        ("median_fill", median_transformer, ["LotFrontage"]),
        ("mode_fill", mode_fill_pipeline, ["Electrical"]),
        ("zero_fill", numerical_missing_transformer, zero_fill_columns),
        ("none_fill", none_fill_pipeline, categorical_none_columns),
        ("catch_all_numeric", catch_all_numeric, remaining_numeric),
        ("catch_all_categorical", catch_all_categorical_pipeline, remaining_categorical)
    ], remainder="passthrough")

    preprocessor.set_output(transform="pandas")

    return preprocessor
