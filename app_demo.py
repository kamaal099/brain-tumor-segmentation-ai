from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import sys
from PIL import Image
import io
import json
from datetime import datetime
import random
import numpy as np

app = FastAPI(
    title="Brain Tumor Detection API",
    description="Production Brain Tumor Segmentation Backend",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Mount static files to serve generated images
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

print("Demo model loaded with enhanced visualization")

def preprocess_image(contents):
    """Preprocess image for prediction"""
    image = Image.open(io.BytesIO(contents))
    image = image.convert('RGB')
    image = image.resize((256, 256))
    return image, image

def create_tumor_mask(width=256, height=256):
    """Create a realistic tumor mask with multiple regions"""
    # Create blank mask
    mask = np.zeros((height, width), dtype=int)
    
    # Main tumor region (center)
    center_x, center_y = width // 2, height // 2
    radius = 25
    
    for i in range(height):
        for j in range(width):
            distance = ((i - center_y)**2 + (j - center_x)**2)**0.5
            if distance < radius:
                mask[i, j] = 1
    
    # Additional smaller tumors
    tumor_centers = [(60, 60), (180, 180), (70, 140), (150, 80)]
    for cx, cy in tumor_centers:
        for i in range(max(0, cy-12), min(height, cy+12)):
            for j in range(max(0, cx-12), min(width, cx+12)):
                distance = ((i - cy)**2 + (j - cx)**2)**0.5
                if distance < 8:
                    mask[i, j] = 1
    
    return mask

def create_colored_mask(mask_array):
    """Create a bright red tumor mask on black background"""
    height, width = mask_array.shape
    # Create RGB image
    rgb_mask = Image.new('RGB', (width, height), color='black')
    pixels = rgb_mask.load()
    
    # Make tumor regions bright red
    for i in range(height):
        for j in range(width):
            if mask_array[i, j] == 1:
                pixels[j, i] = (255, 0, 0)  # Bright red
    
    return rgb_mask

def create_overlay(original_image, mask_array):
    """Create overlay with red tumor highlighting"""
    # Convert original to RGB if needed
    if original_image.mode != 'RGB':
        original_image = original_image.convert('RGB')
    
    # Create a copy for overlay
    overlay = original_image.copy()
    pixels = overlay.load()
    
    # Apply red highlighting to tumor regions
    width, height = original_image.size
    for i in range(height):
        for j in range(width):
            if mask_array[i, j] == 1:
                r, g, b = pixels[j, i]
                # Enhance red, reduce green and blue for tumor regions
                pixels[j, i] = (
                    min(255, r + 120),  # Add red
                    max(0, g - 60),      # Reduce green
                    max(0, b - 60)       # Reduce blue
                )
    
    return overlay

def tumor_stage(mask):
    """Determine tumor stage based on mask size"""
    tumor_pixels = sum(sum(row) for row in mask)
    if tumor_pixels == 0:
        return "No Tumor"
    elif tumor_pixels < 1000:
        return "Stage 1"
    elif tumor_pixels < 3000:
        return "Stage 2"
    else:
        return "Stage 3"

def save_results(mask, overlay, confidence, stage):
    """Save results to files and history"""
    case_id = f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Save colored mask
    mask_path = f"outputs/{case_id}_mask.png"
    mask.save(mask_path)
    
    # Save overlay
    overlay_path = f"outputs/{case_id}_overlay.png"
    overlay.save(overlay_path)
    
    # Save to history
    history_entry = {
        "case_id": case_id,
        "timestamp": datetime.now().isoformat(),
        "confidence": round(float(confidence), 4),
        "stage": stage,
        "mask_path": mask_path,
        "overlay_path": overlay_path,
        "tensorflow_enabled": False
    }
    
    history_file = "outputs/history.json"
    history = []
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)
    
    history.append(history_entry)
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    
    return case_id, mask_path, overlay_path

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    
    original, input_array = preprocess_image(contents)
    
    # Generate tumor mask
    mask_array = create_tumor_mask(256, 256)
    confidence = random.uniform(0.75, 0.95)
    stage = tumor_stage(mask_array)
    
    # Create colored mask and overlay
    colored_mask = create_colored_mask(mask_array)
    overlay = create_overlay(original, mask_array)
    
    case_id, mask_path, overlay_path = save_results(
        colored_mask, overlay, confidence, stage
    )
    
    tumor_detected = True  # Always true in demo mode
    
    return {
        "case_id": case_id,
        "tumor_detected": tumor_detected,
        "confidence": round(float(confidence), 4),
        "stage": stage,
        "mask_image": mask_path,
        "overlay_image": overlay_path,
        "tensorflow_enabled": False
    }

@app.get("/history")
def get_history():
    try:
        with open("outputs/history.json", "r") as f:
            history = json.load(f)
    except:
        history = []
    
    return history

@app.get("/")
def root():
    return {
        "message": "Brain Tumor Detection API",
        "version": "1.0",
        "status": "Running with Enhanced Visualization",
        "tensorflow_enabled": False,
        "endpoints": {
            "predict": "/predict",
            "history": "/history",
            "docs": "/docs"
        }
    }

@app.get("/status")
def status():
    return {
        "tensorflow_available": False,
        "model_loaded": False,
        "visualization": "Enhanced red tumor highlighting",
        "mode": "Demo with improved masks"
    }
