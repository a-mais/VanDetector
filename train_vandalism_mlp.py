"""Treina e audita o MLP especializado em vandalismo e destruição."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="vandalism_mlp_dataset.csv")
    parser.add_argument("--output", default="models/vandalism_destruction_mlp.pkl")
    args = parser.parse_args()

    data = pd.read_csv(args.dataset).dropna()
    feature_names = [name for name in data if name not in {"label", "split", "image"}]
    train = data.loc[data.split == "train"]
    test = data.loc[data.split.isin(["valid", "test"])]
    if train.empty or test.empty or train.label.nunique() != 2:
        raise SystemExit("O dataset deve ter train e valid/test com as duas classes.")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
            early_stopping=True, max_iter=500, random_state=42,
        )),
    ])
    pipeline.fit(train[feature_names], train.label.astype(int))
    prediction = pipeline.predict(test[feature_names])
    matrix = confusion_matrix(test.label.astype(int), prediction, labels=[0, 1])
    report = classification_report(
        test.label.astype(int), prediction, labels=[0, 1],
        target_names=["normal", "vandalismo_ou_destruicao"], zero_division=0,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "feature_columns": feature_names}, output)
    report_path = output.with_suffix(".report.txt")
    report_path.write_text(
        "MLP VANDALISMO E DESTRUICAO - HOLDOUT\n\n" + report + f"\nMatriz [normal, positivo]:\n{matrix}\n",
        encoding="utf-8",
    )
    print(report)
    print(matrix)
    print(f"Modelo: {output}\nRelatório: {report_path}")


if __name__ == "__main__":
    main()
