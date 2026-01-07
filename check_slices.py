from ml.data import process_data
from ml.model import compute_model_metrics, inference


def check_performance_on_slices(df, feature, model, encoder, lb):
    """
    Check model performance on different slices of a categorical feature.
    Results are saved to slice_output.txt.
    """
    slice_results = []
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]

    for value in df[feature].unique():
        df_slice = df[df[feature] == value]
        X_slice, y_slice, _, _ = process_data(
            df_slice,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb
        )

        preds = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)

        result = (f"Feature: {feature} | Value: {value} | "
                  f"Precision: {precision:.2f} | Recall: {recall:.2f} | "
                  f"Fbeta: {fbeta:.2f}")
        slice_results.append(result)

    with open("slice_output.txt", "w") as f:
        for line in slice_results:
            f.write(line + "\n")


if __name__ == "__main__":
    # Example execution logic
    pass
