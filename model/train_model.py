import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# You'll likely use the starter code's 'process_data' function here
# but for a simplified version:

def train_model(data_path):
    df = pd.read_csv(data_path)
    
    # Simple binary target encoding
    df['salary'] = df['salary'].apply(lambda x: 1 if x == ">50K" else 0)
    
    # Separate features and target
    X = df.drop("salary", axis=1)
    y = df["salary"]
    
    # Handle categorical variables (Simplified)
    X = pd.get_dummies(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Save model and the list of columns (important for inference later)
    joblib.dump(model, "model/model.joblib")
    joblib.dump(X.columns.tolist(), "model/encoder_columns.joblib")
    print("Model saved to model/model.joblib")

if __name__ == "__main__":
    train_model("data/census_clean.csv")