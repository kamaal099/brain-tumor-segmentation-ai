import numpy as np
import cv2
import uuid
import os
import json
from datetime import datetime

IMG_SIZE = 240

MASK_FOLDER = "outputs/masks"
OVERLAY_FOLDER = "outputs/overlays"
HISTORY_FILE = "outputs/history.json"


def preprocess_image(file_bytes):
    image = cv2.imdecode(
        np.frombuffer(file_bytes, np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image.astype(np.float32)

    # Z-score normalization
    image = (image - np.mean(image)) / (np.std(image) + 1e-6)

    image_4ch = np.stack([image]*4, axis=-1)
    image_4ch = np.expand_dims(image_4ch, axis=0)

    return image, image_4ch


def postprocess_prediction(prediction):
    mask = np.argmax(prediction, axis=-1)
    confidence = float(np.max(prediction))
    return mask, confidence


def create_overlay(original, mask):
    color_mask = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    color_mask[mask == 1] = [255, 0, 0]    # Red
    color_mask[mask == 2] = [0, 255, 0]    # Green
    color_mask[mask == 3] = [0, 0, 255]    # Blue

    original_rgb = cv2.cvtColor(original.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(original_rgb, 0.7, color_mask, 0.3, 0)

    return overlay


def tumor_stage(mask):
    area = np.sum(mask > 0)

    if area < 500:
        return "Stage I (Small)"
    elif area < 3000:
        return "Stage II (Medium)"
    else:
        return "Stage III (Large)"


def save_results(mask, overlay, confidence, stage):
    case_id = str(uuid.uuid4())

    mask_path = os.path.join(MASK_FOLDER, f"{case_id}.png")
    overlay_path = os.path.join(OVERLAY_FOLDER, f"{case_id}.png")

    cv2.imwrite(mask_path, (mask * 80).astype(np.uint8))
    cv2.imwrite(overlay_path, overlay)

    entry = {
        "case_id": case_id,
        "confidence": confidence,
        "stage": stage,
        "timestamp": datetime.now().isoformat(),
        "mask_path": mask_path,
        "overlay_path": overlay_path
    }

    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            try:
                history = json.load(f)
            except:
                history = []
    else:
        history = []

    history.append(entry)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

    return case_id, mask_path, overlay_path
