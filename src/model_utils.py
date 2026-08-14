from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "hand_dataset_1000.csv"
MODEL_PATH = BASE_DIR / "gesture_model.joblib"


def train_and_save_model():
    dataset = pd.read_csv(DATASET_PATH)

    X = dataset.iloc[:, 1:].values
    Y = dataset.iloc[:, 0].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score, classification_report
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

    joblib.dump({"classifier": classifier, "scaler": scaler}, MODEL_PATH)
    return classifier, scaler


def load_or_train_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_mtime >= DATASET_PATH.stat().st_mtime:
        model = joblib.load(MODEL_PATH)
        return model["classifier"], model["scaler"]

    return train_and_save_model()
