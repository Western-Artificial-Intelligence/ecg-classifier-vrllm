import os
import struct
import base64
import io
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
import numpy as np
import tensorflow as tf

# Import project modules for prediction and Grad-CAM
from src import config
from src.utilities.preprocess import preprocess
from src.utilities.gradcam import make_gradcam_heatmap, save_gradcam_visualization

# Define the base directory for ECG data relative to the project root
# This assumes the backend is run from the project root or similar.
ECG_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/raw/ecgdata")

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

    file_full_path = os.path.join(ECG_DATA_DIR, filename)

    if not os.path.exists(file_full_path):
        raise HTTPException(status_code=404, detail=f"File {filename} not found at {ECG_DATA_DIR}")

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
        
        # Preprocess the record
        full_record_path = os.path.join(config.RAW_DATA_DIR, record_name)
        out = preprocess(full_record_path)
        
        tensors = out["tensors"]
        minutes = out["minutes"]
        skipped = out.get("skipped", [])
        
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
            "skipped": skipped
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/api/gradcam/{filename}")
async def generate_gradcam(filename: str, minute: Optional[int] = Query(None)):
    """
    Generates Grad-CAM visualization for a specific minute of a record.
    If minute is not specified, generates for the most confident prediction.
    """
    record_name = filename.replace(".dat", "")
    
    try:
        # Load model
        model = get_model()
        
        # Preprocess the record
        full_record_path = os.path.join(config.RAW_DATA_DIR, record_name)
        out = preprocess(full_record_path)
        
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
        
        # Generate visualization and convert to base64
        import matplotlib
        matplotlib.use('Agg')  # Use non-GUI backend
        import matplotlib.pyplot as plt
        import tempfile
        
        # Save to temporary file first, then read for base64 encoding
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            save_gradcam_visualization(raw_signal, heatmap, temp_path, alpha=0.4)
            
            # Read the file and convert to base64
            with open(temp_path, 'rb') as f:
                img_base64 = base64.b64encode(f.read()).decode('utf-8')
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return {
            "filename": filename,
            "minute": int(target_minute),
            "image_url": f"data:image/png;base64,{img_base64}",
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
    """
    record_name = filename.replace(".dat", "")
    
    try:
        # Load model
        model = get_model()
        
        # Preprocess the record
        full_record_path = os.path.join(config.RAW_DATA_DIR, record_name)
        out = preprocess(full_record_path)
        
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
            
            # Generate visualization using temp file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            try:
                save_gradcam_visualization(raw_signal, heatmap, temp_path, alpha=0.4)
                with open(temp_path, 'rb') as f:
                    img_base64 = base64.b64encode(f.read()).decode('utf-8')
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            results.append({
                "minute": int(target_minute),
                "image_url": f"data:image/png;base64,{img_base64}",
                "probability": confidence,
                "predicted_class": class_label
            })
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch Grad-CAM generation error: {str(e)}")


@app.get("/api/status")
async def get_status():
    """Simple endpoint to check if the API is running."""
    return {"status": "ok", "message": "ECG Backend API is running."}

