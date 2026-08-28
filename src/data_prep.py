import os
import pandas as pd
import joblib
from paths import DATA_RAW, MODELS_DIR
from preprocessing import clean_raw_data, build_preprocessor

train_df = pd.read_csv(os.path.join(DATA_RAW, "train.csv"))

train_df = train_df.drop(index=1298)

clean_train_df = clean_raw_data(train_df)

preprocessor = build_preprocessor(clean_train_df)

X_transformed = preprocessor.fit_transform(clean_train_df)

joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor2.joblib"))

preprocessor_path = os.path.join(MODELS_DIR, "preprocessor2.joblib")
print(
    f"Preprocessor saved to: {os.path.join(MODELS_DIR, "preprocessor2.joblib")}")
