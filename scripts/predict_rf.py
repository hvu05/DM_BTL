#!/usr/bin/env python3
"""Run inference on a single image using a trained Random Forest model (HOG + Random Forest).

The image goes through exactly the same preprocessing as in training:
resize to 224x224 (direct, or keep aspect ratio + neutral-gray padding) -> grayscale -> HOG.
The probabilities are the Random Forest's own predict_proba (mean of the trees' class votes),
so no softmax over decision margins is needed as for SVM.

    python scripts/predict_rf.py path/to/image.jpg
    python scripts/predict_rf.py path/to/image.jpg --model outputs/rf/rf_padding_model.joblib
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

warnings.filterwarnings("ignore")

import cv2
import joblib
import numpy as np
from skimage.feature import hog

CLASSES = ["billete", "knife", "monedero", "pistol", "smartphone", "tarjeta"]

CLASS_INFO = {
    "knife": {"vi": "Dao / Vũ khí lạnh", "type": "VŨ KHÍ NGUY HIỂM", "is_weapon": True},
    "pistol": {"vi": "Súng ngắn / Vũ khí nóng", "type": "VŨ KHÍ NGUY HIỂM", "is_weapon": True},
    "smartphone": {"vi": "Điện thoại thông minh", "type": "Đồ vật thông thường", "is_weapon": False},
    "tarjeta": {"vi": "Thẻ ngân hàng / Thẻ từ", "type": "Đồ vật thông thường", "is_weapon": False},
    "billete": {"vi": "Tiền giấy / Tiền mặt", "type": "Đồ vật thông thường", "is_weapon": False},
    "monedero": {"vi": "Ví tiền / Bóp cầm tay", "type": "Đồ vật thông thường", "is_weapon": False},
}

HOG_CONFIG = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}
TARGET_SIZE = 224
GRAY_BGR = (128, 128, 128)


def resize_image(img: np.ndarray, branch: str) -> np.ndarray:
    """Same resize as build_resized_branches: 'direct' squashes, 'padding' keeps aspect ratio."""
    h, w = img.shape[:2]
    if branch == "direct":
        interp = cv2.INTER_AREA if (w > TARGET_SIZE or h > TARGET_SIZE) else cv2.INTER_LINEAR
        return cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=interp)
    scale = TARGET_SIZE / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    scaled = cv2.resize(img, (nw, nh), interpolation=interp)
    top, left = (TARGET_SIZE - nh) // 2, (TARGET_SIZE - nw) // 2
    canvas = np.full((TARGET_SIZE, TARGET_SIZE, 3), GRAY_BGR, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = scaled
    return canvas


def extract_hog(img_bgr: np.ndarray, branch: str) -> np.ndarray:
    gray = cv2.cvtColor(resize_image(img_bgr, branch), cv2.COLOR_BGR2GRAY)
    return hog(gray, **HOG_CONFIG).astype(np.float32).reshape(1, -1)


def branch_from_model_path(model_path: Path) -> str:
    return "padding" if "padding" in model_path.name else "direct"


def predict_bgr(img_bgr: np.ndarray, model, branch: str) -> dict:
    feat = extract_hog(img_bgr, branch)
    probs = model.predict_proba(feat)[0]
    results = []
    for idx, prob in zip(model.classes_, probs):
        cname = CLASSES[int(idx)]
        results.append({
            "class": cname,
            "name_vi": CLASS_INFO[cname]["vi"],
            "category": CLASS_INFO[cname]["type"],
            "is_weapon": CLASS_INFO[cname]["is_weapon"],
            "prob": float(prob),
        })
    results.sort(key=lambda x: x["prob"], reverse=True)
    return {
        "predicted_class": results[0]["class"],
        "class_info": CLASS_INFO[results[0]["class"]],
        "top_prediction": results[0],
        "all_probabilities": results,
    }


def predict_image(image_path: Path, model_path: Path, branch: str | None = None) -> dict:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    model = joblib.load(model_path)
    return predict_bgr(img, model, branch or branch_from_model_path(model_path))


def render_progress_bar(prob: float, width: int = 20) -> str:
    filled = int(round(prob * width))
    return "█" * filled + "░" * (width - filled)


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict weapon/object class for an image using Random Forest")
    parser.add_argument("image", type=Path, help="Path to input image file")
    parser.add_argument("--model", type=Path, default=Path("outputs/rf/rf_direct_model.joblib"),
                        help="Path to trained Random Forest .joblib")
    parser.add_argument("--branch", choices=["direct", "padding"], default=None,
                        help="Resize branch the model was trained on (default: read from the model file name)")
    args = parser.parse_args()

    branch = args.branch or branch_from_model_path(args.model)
    res = predict_image(args.image, args.model, branch)
    top = res["top_prediction"]

    print("\n" + "=" * 60)
    print(f" KẾT QUẢ PHÂN LOẠI ẢNH (HOG + RANDOM FOREST, nhánh {branch})")
    print("=" * 60)
    print(f"Ảnh đầu vào      : {args.image}")
    print(f"Dự đoán vật thể  : {top['class'].upper()} ({top['name_vi']})")
    print(f"Nhóm phân loại   : {top['category'].upper()}")
    print(f"Độ tin cậy (Prob): {top['prob'] * 100:.2f}%  (tỷ lệ phiếu bầu của các cây)")
    print("-" * 60)
    print("BẢNG PHÂN BỐ XÁC SUẤT 6 LỚP:")
    for r in res["all_probabilities"]:
        bar = render_progress_bar(r["prob"], width=16)
        tag = "[VŨ KHÍ]" if r["is_weapon"] else "[THƯỜNG]"
        print(f"  {r['class']:<11} ({r['name_vi']:<22}) {tag:<9} | {bar} {r['prob']*100:>5.1f}%")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
