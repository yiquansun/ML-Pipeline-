from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import joblib

def compute_slice_metrics(df, feature, model, encoder_cols):
    """ Outputs performance on slices of a categorical feature. """
    results = []
    for value in df[feature].unique():
        slice_df = df[df[feature] == value]
        
        # Prepare slice for prediction (matching training columns)
        y_true = slice_df['salary'].apply(lambda x: 1 if x == ">50K" else 0)
        X_slice = pd.get_dummies(slice_df.drop("salary", axis=1))
        X_slice = X_slice.reindex(columns=encoder_cols, fill_value=0)
        
        preds = model.predict(X_slice)
        
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        
        results.append(f"{feature} - {value}: Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
    
    with open("slice_output.txt", "a") as f:
        for line in results:
            f.write(line + "\n")

# Usage: Run this for 'education', 'race', 'sex', etc.