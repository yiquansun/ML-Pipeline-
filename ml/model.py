from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    precision = precision_score(y, preds, pos_label=1)
    recall = recall_score(y, preds, pos_label=1)
    fbeta = fbeta_score(y, preds, beta=1, pos_label=1)
    return precision, recall, fbeta


def inference(model, X):
    preds = model.predict(X)
    return preds
