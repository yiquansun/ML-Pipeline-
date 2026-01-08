import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
        X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    if label is not None and label in X.columns:
        y = X[label].values
        X = X.drop([label], axis=1)
    else:
        y = None

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y).flatten()
    else:
        X_categorical = encoder.transform(X_categorical)
        # Only transform y if it's not None and we have a LabelBinarizer
        if y is not None and lb is not None:
            try:
                y = lb.transform(y).flatten()
            except Exception:
                pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb
