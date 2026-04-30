from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
import os
import uuid

app = FastAPI(
    title="Brain Tumor Detection API",
    description="Fixed Brain Tumor Segmentation Backend",
    version="2.0"
)

# ------------------ CORS ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ LOAD MODEL ------------------
print("Loading model...")
model = tf.keras.models.load_model(
    "model/unet_brats2d_final_dice_only.h5",
    compile=False
)
print("Model loaded successfully.")

# ------------------ CREATE OUTPUT FOLDER ------------------
os.makedirs("outputs", exist_ok=True)


# =========================================================
# 🔥 PREPROCESS (FIXED)
# =========================================================
def preprocess_image(contents):
    file_bytes = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (128, 128))

    img = img / 255.0

    # 🔥 simulate 4 MRI modalities
    img_4ch = np.stack([img, img, img, img], axis=-1)

    input_tensor = np.expand_dims(img_4ch, axis=0)

    return img, input_tensor


# =========================================================
# 🔥 POSTPROCESS (FIXED)
# =========================================================
def postprocess_prediction(prediction):
    mask = np.argmax(prediction, axis=-1)
    confidence = float(np.max(prediction))
    return mask, confidence


# =========================================================
# 🔥 OVERLAY (FIXED)
# =========================================================
def create_overlay(image, mask):
    image = (image * 255).astype(np.uint8)

    colored = cv2.applyColorMap((mask * 80).astype(np.uint8), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image, 0.6, colored, 0.4, 0)

    return overlay


# =========================================================
# 🔥 SAVE RESULTS
# =========================================================
def save_results(mask, overlay):
    case_id = str(uuid.uuid4())[:8]

    mask_path = f"outputs/{case_id}_mask.png"
    overlay_path = f"outputs/{case_id}_overlay.png"

    cv2.imwrite(mask_path, (mask * 80).astype(np.uint8))
    cv2.imwrite(overlay_path, overlay)

    return case_id, mask_path, overlay_path


# =========================================================
# 🔥 STAGE DETECTION
# =========================================================
def tumor_stage(mask):
    pixels = np.sum(mask > 0)

    if pixels == 0:
        return "No Tumor"
    elif pixels < 500:
        return "Early Stage"
    elif pixels < 2000:
        return "Moderate Stage"
    else:
        return "Advanced Stage"


# =========================================================
# 🚀 PREDICT API
# =========================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    # preprocess
    original, input_tensor = preprocess_image(contents)

    # prediction
    prediction = model.predict(input_tensor)[0]

    print("Prediction range:", prediction.min(), prediction.max())

    # postprocess
    mask, confidence = postprocess_prediction(prediction)

    print("Mask values:", np.unique(mask))

    # stage
    stage = tumor_stage(mask)

    # overlay
    overlay = create_overlay(original, mask)

    # save
    case_id, mask_path, overlay_path = save_results(mask, overlay)

    tumor_detected = bool(np.sum(mask) > 0)

    return {
        "case_id": case_id,
        "tumor_detected": tumor_detected,
        "confidence": round(confidence, 4),
        "stage": stage,
        "mask_image": mask_path,
        "overlay_image": overlay_path
    }


# =========================================================
# 📜 HISTORY
# =========================================================
@app.get("/history")
def get_history():
    return {"message": "History feature optional"}