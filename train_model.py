import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from ml.data import process_data
from sklearn.ensemble import RandomForestClassifier

# Load cleaned data
data = pd.read_csv("data/census_clean.csv")

# Identify categorical features
cat_features = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]

# Split data
train, test = train_test_split(data, test_size=0.20)

# Process training data
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
joblib.dump(lb, "model/label_binarizer.joblib")
print("Label Binarizer saved successfully!")
# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save artifacts
joblib.dump(model, "model/model.joblib")
joblib.dump(encoder, "model/encoder.joblib")
joblib.dump(lb, "model/lb.joblib")

print("Model and artifacts saved successfully!")