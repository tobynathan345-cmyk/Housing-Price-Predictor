import os
import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from paths import DATA_RAW, MODELS_DIR, PROJECT_ROOT
from preprocessing import clean_raw_data

# Load the training data and the preprocessor
test_df = pd.read_csv(os.path.join(DATA_RAW, "test.csv"))
model = joblib.load(os.path.join(MODELS_DIR, "final_xgb_model.joblib"))

clean_test_df = clean_raw_data(test_df)

predictions = model.predict(clean_test_df)

submission = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": predictions
})

print(submission.head())

submission_path = os.path.join(PROJECT_ROOT, "submission_xgb.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission saved to: {submission_path}")
