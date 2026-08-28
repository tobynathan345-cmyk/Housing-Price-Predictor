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

print(X.info())
