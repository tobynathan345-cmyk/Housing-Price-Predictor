import os
import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from paths import DATA_RAW, MODELS_DIR
from preprocessing import clean_raw_data, build_preprocessor

# Load the training data and the preprocessor
train_df = pd.read_csv(os.path.join(DATA_RAW, "train.csv"))
preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))

# Drop the outlier and clean the training data
train_df = train_df.drop(index=1298)
clean_train_df = clean_raw_data(train_df)

# Separate training data into the target SalePrice and what we are training on
X = clean_train_df.drop(columns=["SalePrice"])
y = clean_train_df["SalePrice"]

# Build our combined pipeline of preprocessing and the model and loop through different n_estimators
lr = 0.1
num_est = 100

for max_depth in [None, 3, 4, 5, 6]:
    xgb_pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", XGBRegressor(n_estimators=num_est,
         learning_rate=lr, max_depth=max_depth, random_state=0))
    ])

    # Run cross validation to get a reliable performance estimate across 5 different train/validation splits
    scores = cross_val_score(xgb_pipeline, X,
                             y, cv=5, scoring="neg_root_mean_squared_error")

    print(
        f"XGBRegressor n_estimators = {num_est}, learning_rate = {lr}, max_depth = {max_depth}: avg RMSE = {-scores.mean():.0f}, fold range = {-scores.max():.0f} - {-scores.min():.0f}")
