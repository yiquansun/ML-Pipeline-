import pandas as pd
import joblib
from ml.data import process_data
from sklearn.metrics import precision_score, recall_score, f1_score

def run_slicing_test():
    # Load data and artifacts
    df = pd.read_csv("data/census_clean.csv")
    model = joblib.load("model/model.joblib")
    encoder = joblib.load("model/encoder.joblib")
    lb = joblib.load("model/lb.joblib")
    
    cat_features = ["workclass", "education", "marital-status", "occupation", 
                    "relationship", "race", "sex", "native-country"]

    slice_output = []
    # Let's slice by 'education' as a primary example
    feature = "education"
    
    for value in df[feature].unique():
        slice_df = df[df[feature] == value]
        
        # Prepare data for this specific slice
        X, y, _, _ = process_data(
            slice_df, categorical_features=cat_features, 
            label="salary", training=False, encoder=encoder, lb=lb
        )
        
        preds = model.predict(X)
        precision = precision_score(y, preds, zero_division=0)
        recall = recall_score(y, preds, zero_division=0)
        f1 = f1_score(y, preds, zero_division=0)
        
        line = f"Slice {feature}={value} | Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f}"
        print(line)
        slice_output.append(line)
    
    # Save to the required file
    with open("slice_output.txt", "w") as f:
        f.write("\n".join(slice_output))

if __name__ == "__main__":
    run_slicing_test()