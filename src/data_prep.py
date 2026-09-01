import os
import pandas as pd
import joblib
from paths import DATA_RAW, MODELS_DIR
from preprocessing import clean_raw_data, build_preprocessor

train_df = pd.read_csv(os.path.join(DATA_RAW, "train.csv"))

train_df = train_df.drop(index=1298)

X = train_df.drop(columns=["SalePrice"])

X_clean = clean_raw_data(X)

preprocessor = build_preprocessor(X_clean)

X_transformed = preprocessor.fit_transform(X_clean)

preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
joblib.dump(preprocessor, preprocessor_path)
print(
    f"Preprocessor saved to: {preprocessor_path}")
