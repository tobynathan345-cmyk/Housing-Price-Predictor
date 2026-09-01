import os
import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import Ridge
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

# Build the pipeline and cross validation for different values of alpha to see which performs the best
"""
for alpha in [0.1, 1, 10, 50, 100]:
    ridge_pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", Ridge(alpha=alpha))
    ])
    scores = cross_val_score(ridge_pipeline, X, y, cv=5,
                             scoring="neg_root_mean_squared_error")
    print(
        f"Ridge alpha = {alpha}: avg RMSE = {-1*scores.mean():.0f}, fold range = {-1*scores.max():.0f} - {-1*scores.min():.0f}")
"""
# Best was alpha = 10

final_ridge_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", Ridge(alpha=10))
])

model = final_ridge_pipeline.fit(X, y)

model_path = os.path.join(MODELS_DIR, "final_ridge_model.joblib")
joblib.dump(model, model_path)
print(f"Final ridge model saved to: {model_path}")
