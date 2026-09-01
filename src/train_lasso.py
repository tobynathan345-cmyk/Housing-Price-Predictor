import os
import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from paths import DATA_RAW, MODELS_DIR
from preprocessing import clean_raw_data, build_preprocessor

# Load the training data
train_df = pd.read_csv(os.path.join(DATA_RAW, "train.csv"))

# Drop the outlier and clean the training data
train_df = train_df.drop(index=1298)
clean_train_df = clean_raw_data(train_df)

# Separate training data into the target SalePrice and what we are training on and build preprocessor
X = clean_train_df.drop(columns=["SalePrice"])
y = clean_train_df["SalePrice"]

preprocessor = build_preprocessor(X)

# Build the pipeline and cross validation for different values of alpha to see which performs the best
for alpha in [100, 150, 200, 250, 300]:
    lasso_pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", Lasso(alpha=alpha))
    ])
    scores = cross_val_score(lasso_pipeline, X, y, cv=5,
                             scoring="neg_root_mean_squared_error")
    print(
        f"Lasso alpha = {alpha}: avg RMSE = {-1*scores.mean():.0f}, fold range = {-1*scores.max():.0f} - {-1*scores.min():.0f}")

# Best was alpha = 200
