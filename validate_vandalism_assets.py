"""Valida os assets rotulados usando YOLO + MLP de vandalismo.

O teste é de nível de vídeo: um asset positivo precisa conter pelo menos um
alerta e um asset normal não pode conter alerta. O comando encerra com erro
quando qualquer caso falhar, impedindo regressões silenciosas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import joblib
import pandas as pd
from ultralytics import YOLO

from src.roboflow_features import ultralytics_result_to_features


def probability(bundle, features: dict[str, float]) -> float:
    row = pd.DataFrame([{name: features.get(name, 0.0) for name in bundle["feature_columns"]}])
    pipeline = bundle["pipeline"]
    positive_index = list(pipeline.classes_).index(1)
    return float(pipeline.predict_proba(row)[0][positive_index])


def evaluate(video: Path, expected: str, yolo: YOLO, bundle, sample_seconds: float, threshold: float) -> dict:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Não foi possível abrir {video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps * sample_seconds))
    best_probability = 0.0
    first_hit = None
    samples = 0
    for index in range(0, frame_count, step):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        result = yolo(frame, verbose=False)[0]
        class_names = [result.names[i] for i in sorted(result.names)]
        score = probability(bundle, ultralytics_result_to_features(result, class_names))
        samples += 1
        if score > best_probability:
            best_probability = score
        if first_hit is None and score >= threshold:
            first_hit = round(index / fps, 2)
    capture.release()
    predicted = "vandalism" if first_hit is not None else "normal"
    return {
        "video": str(video), "expected": expected, "predicted": predicted,
        "passed": predicted == expected, "best_probability": round(best_probability, 4),
        "first_alert_s": first_hit if first_hit is not None else "", "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="asset_cases.json")
    parser.add_argument("--weights", default="runs/detect/vandalism2/weights/best.pt")
    parser.add_argument("--model", default="models/vandalism_mlp.pkl")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--sample-seconds", type=float, default=3.0)
    parser.add_argument("--output", default="outputs/validation/vandalism_assets.csv")
    args = parser.parse_args()
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    yolo, bundle = YOLO(args.weights), joblib.load(args.model)
    rows = [evaluate(Path(case["video"]), case["expected"], yolo, bundle,
                     args.sample_seconds, args.threshold) for case in cases]
    result = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"\nCasos aprovados: {int(result.passed.sum())}/{len(result)}")
    print(f"Relatório: {output}")
    if not result.passed.all():
        raise SystemExit("A validação dos assets falhou.")


if __name__ == "__main__":
    main()
