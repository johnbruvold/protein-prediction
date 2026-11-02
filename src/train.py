"""Reproducible training pipeline for the enzyme-class classifiers.

data/ and models/ are intentionally left out of version control (the raw
sequence data is hundreds of MB and the trained models aren't much smaller),
so run this script locally to regenerate everything the Streamlit app needs:

    python -m src.train

It fits a single StandardScaler on the training split and saves it alongside
the models (models/scaler.pkl) so that inference on a brand-new sequence is
scaled the same way the training data was — scaling a lone sample on its own
has zero variance and collapses every feature to zero, which is why a shared,
pre-fit scaler has to ship with the models rather than being refit per call.
"""

from pathlib import Path
from pickle import dump

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.featurize import extract_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_PER_CLASS = 2000
RANDOM_STATE = 42

ENZYME_CLASSES = [
    "Oxidoreductases",
    "Transferases",
    "Hydrolases",
    "Lyases",
    "Isomerases",
    "Ligases",
    "Translocases",
]


def load_dataset() -> pd.DataFrame:
    frames = []
    for ec in ENZYME_CLASSES:
        path = PROJECT_ROOT / "data" / "CleanedSequences" / f"{ec}Cleaned.csv"
        df = pd.read_csv(path).sample(SAMPLES_PER_CLASS, random_state=RANDOM_STATE)
        df["Label"] = ec
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    print("Loading sequences...")
    dataset = load_dataset()

    print("Extracting features...")
    features = extract_features(dataset["Sequence"])
    labels = dataset["Label"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.33,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Random Forest...")
    rfc = RandomForestClassifier(random_state=RANDOM_STATE)
    rfc.fit(X_train_scaled, y_train)
    rfc_acc = accuracy_score(y_test, rfc.predict(X_test_scaled))
    print(f"Random Forest test accuracy: {100 * rfc_acc:.2f}%")

    print("Training SVM (this is the slow one)...")
    svc = SVC(kernel="rbf", random_state=RANDOM_STATE)
    svc.fit(X_train_scaled, y_train)
    svc_acc = accuracy_score(y_test, svc.predict(X_test_scaled))
    print(f"SVM test accuracy: {100 * svc_acc:.2f}%")

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    with open(models_dir / "RandomForestClassifier.pkl", "wb") as f:
        dump(rfc, f, protocol=5)
    with open(models_dir / "SupportVectorClassifier.pkl", "wb") as f:
        dump(svc, f, protocol=5)
    with open(models_dir / "scaler.pkl", "wb") as f:
        dump(scaler, f, protocol=5)
    print(f"Saved models + scaler to {models_dir}")


if __name__ == "__main__":
    main()
