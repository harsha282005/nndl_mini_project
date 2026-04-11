"""
api.py — FastAPI backend serving the Fashion-MNIST Deep Learning Models
"""

import os
import sys
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image

# ── Import Project Config ───────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import CLASS_NAMES, MODELS_DIR

app = FastAPI(title="Fashion-MNIST Product Discovery API")

# Allow React local dev to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Models at Startup ────────────────────────────────────────────────────
ann_model, cnn_model = None, None

@app.on_event("startup")
def load_models():
    global ann_model, cnn_model
    ann_path = os.path.join(MODELS_DIR, "ANN_final.keras")
    cnn_path = os.path.join(MODELS_DIR, "CNN_final.keras")
    
    if os.path.exists(ann_path):
        ann_model = tf.keras.models.load_model(ann_path)
        print("[INFO] ANN Model loaded successfully at startup.")
    else:
        print(f"[WARN] ANN Model not found at {ann_path}. Will attempt lazy loading later.")
        
    if os.path.exists(cnn_path):
        cnn_model = tf.keras.models.load_model(cnn_path)
        print("[INFO] CNN Model loaded successfully at startup.")
    else:
        print(f"[WARN] CNN Model not found at {cnn_path}. Will attempt lazy loading later.")

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"message": "Fashion-MNIST Inference API is running."}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...), model_choice: str = Form(...)):
    """Receives a 28x28 grayscale image and returns predictions."""
    if not file:
        return {"error": "No file uploaded."}
    
    # Process image into 28x28 array
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("L").resize((28, 28))
    img_array = np.array(image).astype("float32") / 255.0
    
    # Fashion-MNIST expects a black background with white objects. 
    # Real world images often have light backgrounds. We can do a 
    # heuristic check on the corners to see if we should invert it.
    corners_avg = (img_array[0,0] + img_array[0,27] + img_array[27,0] + img_array[27,27]) / 4.0
    if corners_avg > 0.5:
        # Background is light, invert the colors
        img_array = 1.0 - img_array
    
    # Select and check/lazy-load requested model
    global ann_model, cnn_model
    model = None
    
    if model_choice.upper() == "CNN":
        if cnn_model is None:
            cnn_path = os.path.join(MODELS_DIR, "CNN_final.keras")
            if os.path.exists(cnn_path):
                print("[INFO] Lazy loading CNN Model...")
                cnn_model = tf.keras.models.load_model(cnn_path)
        
        model = cnn_model
        input_data = img_array.reshape(1, 28, 28, 1)
        
    else:  # Default to ANN
        if ann_model is None:
            ann_path = os.path.join(MODELS_DIR, "ANN_final.keras")
            if os.path.exists(ann_path):
                print("[INFO] Lazy loading ANN Model...")
                ann_model = tf.keras.models.load_model(ann_path)
                
        model = ann_model
        input_data = img_array.reshape(1, 784)
        
    if model is None:
        print(f"[ERROR] Inference failed: {model_choice} model is not available natively or on disk.")
        return {"error": f"{model_choice} model is not available. Please train it first and save to /models."}
    
    # Predict
    probabilities = model.predict(input_data)[0].tolist()
    
    # Map back to string class names
    confidence_data = []
    for i, prob in enumerate(probabilities):
        confidence_data.append({
            "className": CLASS_NAMES[i],
            "probability": float(prob)
        })
        
    # Sort for best 3
    top_3 = sorted(confidence_data, key=lambda x: x["probability"], reverse=True)[:3]
    top_prediction = top_3[0]["className"]
    
    return {
        "model": model_choice,
        "top_prediction": top_prediction,
        "top_3": top_3,
        "confidence_distribution": confidence_data
    }
