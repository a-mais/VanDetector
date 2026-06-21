"""Features determinísticas para o MLP de vandalismo a partir de caixas YOLO."""

from __future__ import annotations

import numpy as np


BASE_COLUMNS = [
    "total_detections", "max_area", "sum_area", "mean_area", "max_width",
    "max_height", "mean_center_x", "mean_center_y",
]


def feature_columns(class_names: list[str]) -> list[str]:
    return BASE_COLUMNS + [f"count_{name}" for name in class_names]


def boxes_to_features(boxes, class_names: list[str]) -> dict[str, float]:
    """Converte caixas ``(classe, x, y, largura, altura, confiança)`` em features."""
    areas = [width * height for _, _, _, width, height, _ in boxes]
    widths = [width for _, _, _, width, _, _ in boxes]
    heights = [height for _, _, _, _, height, _ in boxes]
    centers_x = [x for _, x, _, _, _, _ in boxes]
    centers_y = [y for _, _, y, _, _, _ in boxes]
    features = {
        "total_detections": float(len(boxes)),
        "max_area": float(max(areas, default=0.0)),
        "sum_area": float(sum(areas)),
        "mean_area": float(np.mean(areas)) if areas else 0.0,
        "max_width": float(max(widths, default=0.0)),
        "max_height": float(max(heights, default=0.0)),
        "mean_center_x": float(np.mean(centers_x)) if centers_x else 0.0,
        "mean_center_y": float(np.mean(centers_y)) if centers_y else 0.0,
    }
    counts = {f"count_{name}": 0.0 for name in class_names}
    for class_id, *_ in boxes:
        if 0 <= class_id < len(class_names):
            counts[f"count_{class_names[class_id]}"] += 1.0
    return {**features, **counts}


def read_yolo_label_file(path):
    """Lê um arquivo YOLO, retornando caixas com confiança 1.0 quando ausente."""
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        boxes.append((*map(float, values[:5]), float(values[5]) if len(values) > 5 else 1.0))
    return [(int(class_id), x, y, width, height, confidence)
            for class_id, x, y, width, height, confidence in boxes]


def ultralytics_result_to_features(result, class_names: list[str]) -> dict[str, float]:
    """Extrai as mesmas features de uma predição YOLO em tempo de execução."""
    boxes = []
    for box in result.boxes:
        x, y, width, height = box.xywhn[0].tolist()
        boxes.append((int(box.cls), float(x), float(y), float(width), float(height), float(box.conf)))
    return boxes_to_features(boxes, class_names)


def remapped_ultralytics_result_to_features(result, class_names: list[str], class_map: dict[str, str]) -> dict[str, float]:
    """Extrai features em uma taxonomia comum para pesos YOLO diferentes."""
    target_ids = {name: index for index, name in enumerate(class_names)}
    boxes = []
    for box in result.boxes:
        source_name = result.names[int(box.cls)]
        target_name = class_map.get(source_name)
        if target_name not in target_ids:
            continue
        x, y, width, height = box.xywhn[0].tolist()
        boxes.append((target_ids[target_name], float(x), float(y), float(width), float(height), float(box.conf)))
    return boxes_to_features(boxes, class_names)
