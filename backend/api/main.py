import os
import struct
import base64
import io
import json
import time
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Optional
import numpy as np
import tensorflow as tf

# Import project modules for prediction and Grad-CAM
from backend.src import config
from backend.src import agent
from backend.src.utilities.preprocess import preprocess_with_cache
from backend.src.utilities.gradcam import make_gradcam_heatmap, save_gradcam_visualization


# Define Grad-CAM images directory
GRADCAM_IMAGES_DIR = os.path.join(config.PROCESSED_DATA_DIR, "gradcam_images")
os.makedirs(GRADCAM_IMAGES_DIR, exist_ok=True)

app = FastAPI()

# Global model cache - loaded at startup
_model = None

def get_model():
    """Load and cache the trained model."""
    global _model
    if _model is None:
        model_path = os.path.join(config.MODELS_DIR, "model.final.keras")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=500, detail=f"Model not found at {model_path}")
        _model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from: {model_path}")
    return _model

# Configure CORS to allow the React frontend to access the API
origins = [
    "http://localhost",
    "http://localhost:5173",  # Default Vite development server port
    "http://127.0.0.1:5173",
    # Add other frontend origins if necessary, especially for Docker setup
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving Grad-CAM images
# Custom StaticFiles class to add cache headers
class CachedStaticFiles(StaticFiles):
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    headers = dict(message.get("headers", []))
                    headers[b"cache-control"] = b"public, max-age=31536000, immutable"
                    message["headers"] = list(headers.items())
                await send(message)
            await super().__call__(scope, receive, send_wrapper)
        else:
            await super().__call__(scope, receive, send)

app.mount("/gradcam_images", CachedStaticFiles(directory=GRADCAM_IMAGES_DIR), name="gradcam_images")

# Helper function to read .dat files
# Assuming 16-bit integer samples
def read_ecg_data(file_path: str) -> List[int]:
    """Reads ECG data from a .dat file (assuming 16-bit integer samples)."""
    samples = []
    try:
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(2)  # Read 2 bytes for a 16-bit integer
                if not chunk:
                    break
                # Unpack as a signed short (16-bit integer)
                sample = struct.unpack('<h', chunk)[0] # '<h' for little-endian signed short
                samples.append(sample)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="ECG data file not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading ECG data: {e}")
    return samples

@app.get("/api/ecg_data/{filename}")
async def get_ecg_data(filename: str):
    """
    Returns ECG data for a specified .dat file.
    """
    if not filename.endswith(".dat"):
        raise HTTPException(status_code=400, detail="Only .dat files are supported.")

    file_full_path = os.path.join(config.RAW_DATA_DIR, filename)

    if not os.path.exists(file_full_path):
        raise HTTPException(status_code=404, detail=f"File {filename} not found at {config.RAW_DATA_DIR}")

    ecg_samples = read_ecg_data(file_full_path)
    return {"filename": filename, "data": ecg_samples}

@app.post("/api/predict/{filename}")
async def predict_apnea(filename: str):
    """
    Makes predictions on a single ECG record.
    Returns minute-by-minute apnea probabilities.
    """
    # Remove .dat extension if present to get record name
    record_name = filename.replace(".dat", "")
    
    try:
        # Load model
        model = get_model()
        
        # Preprocess the record (with caching for performance)
        full_record_path = os.path.join(config.RAW_DATA_DIR, record_name)
        out = preprocess_with_cache(full_record_path)
        
        tensors = out["tensors"]
        minutes = out["minutes"]
        skipped = out.get("skipped", [])
        stats = out.get("stats", {})  # NEW - extract physiological stats
        
        if tensors.shape[0] == 0:
            return {
                "filename": filename,
                "predictions": [],
                "skipped": skipped,
                "message": "No valid signal segments found for prediction"
            }
        
        # Make predictions
        preds = model.predict(tensors)
        
        # Format predictions as list of {minute, probability}
        predictions = []
        for minute, pred in zip(minutes, preds):
            p_flat = np.asarray(pred).ravel()
            prob_apnea = float(p_flat[1]) if p_flat.size == 2 else float(p_flat[0])
            predictions.append({
                "minute": int(minute),
                "probability": prob_apnea
            })
        
        return {
            "filename": filename,
            "predictions": predictions,
            "skipped": skipped,
            "stats": stats  # NEW - include physiological stats in response
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/api/gradcam/{filename}")
async def generate_gradcam(filename: str, minute: Optional[int] = Query(None)):
    """
    Generates Grad-CAM visualization for a specific minute of a record.
    If minute is not specified, generates for the most confident prediction.
    Saves images to disk instead of returning base64.
    """
    record_name = filename.replace(".dat", "")
    
    # Create patient directory
    patient_dir = os.path.join(GRADCAM_IMAGES_DIR, record_name)
    os.makedirs(patient_dir, exist_ok=True)
    
    try:
        # Load model
        model = get_model()
        
        # Preprocess the record (with caching - reuses results from predict endpoint)
        full_record_path = os.path.join(config.RAW_DATA_DIR, record_name)
        out = preprocess_with_cache(full_record_path)
        
        tensors = out["tensors"]
        minutes = out["minutes"]
        raw_segments = out.get("raw_segments", [])
        
        if tensors.shape[0] == 0:
            raise HTTPException(status_code=404, detail="No valid signal segments found")
        
        # Make predictions to get probabilities
        preds = model.predict(tensors)
        
        # Find the target minute
        if minute is None:
            # Use most confident prediction
            max_probs = np.max(preds, axis=1)
            target_idx = np.argmax(max_probs)
        else:
            # Find index of specified minute
            try:
                target_idx = minutes.index(minute)
            except ValueError:
                raise HTTPException(status_code=404, detail=f"Minute {minute} not found in processed segments")
        
        target_minute = minutes[target_idx]
        prediction_prob = preds[target_idx]
        predicted_class = int(np.argmax(prediction_prob))
        class_label = "Apnea" if predicted_class == 1 else "Non-Apnea"
        confidence = float(prediction_prob[predicted_class])
        
        # Check if image already exists on disk
        image_path = os.path.join(patient_dir, f"{target_minute}.png")
        metadata_path = os.path.join(patient_dir, f"{target_minute}.json")
        
        if os.path.exists(image_path) and os.path.exists(metadata_path):
            # Load existing metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return {
                "filename": filename,
                "minute": int(target_minute),
                "image_url": f"http://localhost:8000/gradcam_images/{record_name}/{target_minute}.png",
                "probability": metadata["probability"],
                "predicted_class": metadata["predicted_class"]
            }
        
        # Generate Grad-CAM heatmap
        input_tensor = np.expand_dims(tensors[target_idx], axis=0)
        
        try:
            heatmap = make_gradcam_heatmap(input_tensor, model, "last_conv_layer")
        except Exception as e:
            # Fallback: try to find any Conv1D layer
            conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv1D)]
            if not conv_layers:
                raise HTTPException(status_code=500, detail="No Conv1D layers found in model")
            last_conv_name = conv_layers[-1].name
            heatmap = make_gradcam_heatmap(input_tensor, model, last_conv_name)
        
        # Get raw signal for this segment
        raw_signal = raw_segments[target_idx] if target_idx < len(raw_segments) else None
        
        if raw_signal is None:
            raise HTTPException(status_code=500, detail="Raw signal segment not available")
        
        # Generate visualization and save to disk
        import matplotlib
        matplotlib.use('Agg')  # Use non-GUI backend
        
        save_gradcam_visualization(raw_signal, heatmap, image_path, alpha=0.4)
        
        # Save metadata
        metadata = {
            "probability": float(confidence),
            "predicted_class": class_label,
            "timestamp": time.time()
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        return {
            "filename": filename,
            "minute": int(target_minute),
            "image_url": f"http://localhost:8000/gradcam_images/{record_name}/{target_minute}.png",
            "probability": confidence,
            "predicted_class": class_label
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation error: {str(e)}")


@app.post("/api/gradcam/batch/{filename}")
async def generate_gradcam_batch(filename: str, count: int = Query(default=3)):
    """
    Generates Grad-CAM visualizations for the top N most confident predictions.
    Saves images to disk instead of returning base64.
    """
    record_name = filename.replace(".dat", "")
    
    # Create patient directory
    patient_dir = os.path.join(GRADCAM_IMAGES_DIR, record_name)
    os.makedirs(patient_dir, exist_ok=True)
    
    try:
        # Load model
        model = get_model()
        
        # Preprocess the record (with caching - reuses results from predict endpoint)
        full_record_path = os.path.join(config.RAW_DATA_DIR, record_name)
        out = preprocess_with_cache(full_record_path)
        
        tensors = out["tensors"]
        minutes = out["minutes"]
        raw_segments = out.get("raw_segments", [])
        
        if tensors.shape[0] == 0:
            raise HTTPException(status_code=404, detail="No valid signal segments found")
        
        # Make predictions
        preds = model.predict(tensors)
        
        # Get top N confident predictions
        max_probs = np.max(preds, axis=1)
        top_indices = np.argsort(max_probs)[-count:][::-1]
        
        results = []
        
        import matplotlib
        matplotlib.use('Agg')
        
        for idx in top_indices:
            target_minute = minutes[idx]
            prediction_prob = preds[idx]
            predicted_class = int(np.argmax(prediction_prob))
            class_label = "Apnea" if predicted_class == 1 else "Non-Apnea"
            confidence = float(prediction_prob[predicted_class])
            
            # Check if image already exists
            image_path = os.path.join(patient_dir, f"{target_minute}.png")
            metadata_path = os.path.join(patient_dir, f"{target_minute}.json")
            
            if os.path.exists(image_path) and os.path.exists(metadata_path):
                # Load existing metadata
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                results.append({
                    "minute": int(target_minute),
                    "image_url": f"http://localhost:8000/gradcam_images/{record_name}/{target_minute}.png",
                    "probability": metadata["probability"],
                    "predicted_class": metadata["predicted_class"]
                })
                continue
            
            # Generate Grad-CAM
            input_tensor = np.expand_dims(tensors[idx], axis=0)
            
            try:
                heatmap = make_gradcam_heatmap(input_tensor, model, "last_conv_layer")
            except:
                conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv1D)]
                if conv_layers:
                    heatmap = make_gradcam_heatmap(input_tensor, model, conv_layers[-1].name)
                else:
                    continue
            
            raw_signal = raw_segments[idx] if idx < len(raw_segments) else None
            if raw_signal is None:
                continue
            
            # Save visualization to disk
            save_gradcam_visualization(raw_signal, heatmap, image_path, alpha=0.4)
            
            # Save metadata
            metadata = {
                "probability": float(confidence),
                "predicted_class": class_label,
                "timestamp": time.time()
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            results.append({
                "minute": int(target_minute),
                "image_url": f"http://localhost:8000/gradcam_images/{record_name}/{target_minute}.png",
                "probability": confidence,
                "predicted_class": class_label
            })
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch Grad-CAM generation error: {str(e)}")


@app.get("/api/gradcam/list/{filename}")
async def list_gradcam_images(filename: str):
    """
    Returns list of all available Grad-CAM images for a recording.
    Fast operation - just reads directory and metadata files.
    """
    record_name = filename.replace(".dat", "")
    patient_dir = os.path.join(GRADCAM_IMAGES_DIR, record_name)
    
    if not os.path.exists(patient_dir):
        return {"images": []}
    
    images = []
    for file in os.listdir(patient_dir):
        if file.endswith('.png'):
            minute = int(file.replace('.png', ''))
            metadata_file = os.path.join(patient_dir, f"{minute}.json")
            
            # Load metadata
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {"probability": 0.0, "predicted_class": "Unknown"}
            
            images.append({
                "minute": minute,
                "image_url": f"http://localhost:8000/gradcam_images/{record_name}/{minute}.png",
                "probability": metadata.get("probability", 0.0),
                "predicted_class": metadata.get("predicted_class", "Unknown")
            })
    
    # Sort by minute
    images.sort(key=lambda x: x["minute"])
    
    return {"images": images}


@app.get("/api/status")
async def get_status():
    """Simple endpoint to check if the API is running."""
    return {"status": "ok", "message": "ECG Backend API is running."}


@app.post("/api/agent/analyze/{filename}")
async def analyze_record_with_agent(filename: str):
    """
    Generate one consolidated agent analysis for chat consumption.
    Always returns 200 with either Gemini analysis or a fallback message.
    """
    record_name = filename.replace(".dat", "")
    normalized_filename = f"{record_name}.dat"

    try:
        result = await agent.generate_chat_analysis_for_record(record_name, visualize_count=3)
        return {
            "filename": normalized_filename,
            "analysis": result["analysis"],
            "source": "gemini",
            "meta": result.get("meta", {"analyzed_minutes": 0}),
        }
    except Exception as e:
        fallback_message = (
            f"Analysis is currently unavailable for `{record_name}`. "
            f"Predictions completed, but the agent could not produce a full Grad-CAM explanation. "
            f"Reason: {str(e)}"
        )
        return {
            "filename": normalized_filename,
            "analysis": fallback_message,
            "source": "fallback",
            "meta": {"analyzed_minutes": 0},
        }

