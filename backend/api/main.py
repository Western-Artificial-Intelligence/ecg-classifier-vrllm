import os
import struct
import base64
import io
import json
import time
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

# Import project modules for prediction and Grad-CAM
from backend.src import config
from backend.src import agent
from backend.src.utilities.preprocess import preprocess_with_cache
from backend.src.utilities.gradcam import make_gradcam_heatmap, save_gradcam_visualization
from backend.src.database import get_db_manager


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

# Pydantic models for request/response validation
class PatientCreate(BaseModel):
    name: str
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None

class PatientUpdate(BaseModel):
    name: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None

class RecordCreate(BaseModel):
    patient_id: int
    record_name: str
    file_path: str
    duration_minutes: Optional[float] = None
    sample_rate_hz: int = 100
    recorded_at: Optional[str] = None
    notes: Optional[str] = None

# Optional: Initialize database manager (non-blocking if DB doesn't exist)
try:
    db_manager = get_db_manager()
    print(f"Database manager initialized successfully")
except Exception as e:
    db_manager = None
    print(f"Warning: Database not available: {e}")

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

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

# ============================================================================
# PATIENT ENDPOINTS
# ============================================================================

@app.get("/api/patients")
async def get_all_patients():
    """Get all patients from the database."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        patients = db_manager.get_all_patients()
        return {"patients": patients}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch patients: {str(e)}")

@app.get("/api/patients/{patient_id}")
async def get_patient(patient_id: int):
    """Get a specific patient by ID."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        patient = db_manager.get_patient(patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
        return patient
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch patient: {str(e)}")

@app.post("/api/patients")
async def create_patient(patient: PatientCreate):
    """Create a new patient."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        patient_id = db_manager.add_patient(
            name=patient.name,
            date_of_birth=patient.date_of_birth,
            gender=patient.gender,
            weight_kg=patient.weight_kg,
            height_cm=patient.height_cm
        )
        
        # Return the newly created patient
        new_patient = db_manager.get_patient(patient_id)
        return new_patient
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create patient: {str(e)}")

@app.put("/api/patients/{patient_id}")
async def update_patient(patient_id: int, patient_update: PatientUpdate):
    """Update an existing patient."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Check if patient exists
        existing_patient = db_manager.get_patient(patient_id)
        if not existing_patient:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
        
        # Update patient
        db_manager.update_patient(
            patient_id=patient_id,
            name=patient_update.name,
            date_of_birth=patient_update.date_of_birth,
            gender=patient_update.gender,
            weight_kg=patient_update.weight_kg,
            height_cm=patient_update.height_cm
        )
        
        # Return updated patient
        updated_patient = db_manager.get_patient(patient_id)
        return updated_patient
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update patient: {str(e)}")

@app.get("/api/patients/{patient_id}/records")
async def get_patient_records(patient_id: int):
    """Get all ECG recordings for a specific patient."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Check if patient exists
        patient = db_manager.get_patient(patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
        
        records = db_manager.get_patient_records(patient_id)
        return {"records": records}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch records: {str(e)}")

# ============================================================================
# RECORD ENDPOINTS
# ============================================================================

@app.get("/api/records/{record_id}")
async def get_record(record_id: int):
    """Get a specific ECG record by ID."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        record = db_manager.get_record(record_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"Record {record_id} not found")
        return record
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch record: {str(e)}")

@app.get("/api/records/{record_id}/predictions")
async def get_record_predictions(record_id: int):
    """Get predictions for a specific record."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Check if record exists
        record = db_manager.get_record(record_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"Record {record_id} not found")
        
        predictions = db_manager.get_predictions(record_id)
        if not predictions:
            return {"predictions": None, "message": "No predictions found for this record"}
        
        return predictions
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch predictions: {str(e)}")

@app.get("/api/records/{record_id}/metrics")
async def get_record_metrics(record_id: int):
    """Get physiological metrics for a specific record."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Check if record exists
        record = db_manager.get_record(record_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"Record {record_id} not found")
        
        metrics = db_manager.get_metrics(record_id)
        if not metrics:
            return {"metrics": None, "message": "No metrics found for this record"}
        
        return metrics
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch metrics: {str(e)}")

@app.get("/api/records/{record_id}/gradcams")
async def get_record_gradcams(record_id: int):
    """Get all Grad-CAM images metadata for a specific record."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Check if record exists
        record = db_manager.get_record(record_id)
        if not record:
            raise HTTPException(status_code=404, detail=f"Record {record_id} not found")
        
        gradcams = db_manager.get_gradcam_images(record_id)
        return {"gradcams": gradcams}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Grad-CAM images: {str(e)}")

@app.get("/api/files/available")
async def get_available_files():
    """Get list of available .dat files in the raw data directory."""
    try:
        raw_data_dir = config.RAW_DATA_DIR
        if not os.path.exists(raw_data_dir):
            return {"files": []}
        
        files = []
        for filename in os.listdir(raw_data_dir):
            if filename.endswith('.dat'):
                file_path = os.path.join(raw_data_dir, filename)
                file_size = os.path.getsize(file_path)
                file_mtime = os.path.getmtime(file_path)
                
                # Check if already in database
                record_name = filename.replace('.dat', '')
                is_linked = False
                if db_manager:
                    try:
                        existing = db_manager.get_record_by_name(record_name)
                        is_linked = existing is not None
                    except:
                        pass
                
                files.append({
                    "filename": filename,
                    "record_name": record_name,
                    "size": file_size,
                    "modified": file_mtime,
                    "is_linked": is_linked
                })
        
        # Sort by modified date (newest first)
        files.sort(key=lambda x: x['modified'], reverse=True)
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")

@app.post("/api/records")
async def create_record(record: RecordCreate):
    """Create a new record linking a patient to an existing .dat file."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Check if patient exists
        patient = db_manager.get_patient(record.patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient {record.patient_id} not found")
        
        # Check if record name already exists
        existing = db_manager.get_record_by_name(record.record_name)
        if existing:
            raise HTTPException(status_code=400, detail=f"Record {record.record_name} already exists")
        
        # Construct full file path using config
        # If filename includes .dat, use record_name, otherwise append .dat
        if record.record_name.endswith('.dat'):
            filename = record.record_name
            record_name_clean = record.record_name.replace('.dat', '')
        else:
            filename = f"{record.record_name}.dat"
            record_name_clean = record.record_name
        
        full_file_path = os.path.join(config.RAW_DATA_DIR, filename)
        
        # Verify file exists
        if not os.path.exists(full_file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {filename} (expected at {full_file_path})")
        
        # Create record with the full path
        record_id = db_manager.add_record(
            patient_id=record.patient_id,
            record_name=record_name_clean,
            file_path=full_file_path,
            duration_minutes=record.duration_minutes,
            sample_rate_hz=record.sample_rate_hz,
            recorded_at=record.recorded_at,
            notes=record.notes
        )
        
        # Return the newly created record
        new_record = db_manager.get_record(record_id)
        return new_record
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create record: {str(e)}")

# ============================================================================
# ECG DATA & ANALYSIS ENDPOINTS
# ============================================================================

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
        
        # Optional: Log to database if available
        if db_manager:
            try:
                # Check if record exists
                record = db_manager.get_record_by_name(record_name)
                
                # Only save predictions/metrics if record exists in database
                # Records should be created through PatientManagement linking
                if not record:
                    print(f"Record {record_name} not in database, skipping database save")
                else:
                    record_id = record['id']
                    
                    # Check if predictions already exist for this record
                    existing_preds = db_manager.get_predictions(record_id)
                    if not existing_preds:
                        # Save predictions to database only if they don't exist
                        db_manager.save_predictions(record_id, predictions)
                        print(f"Saved predictions for {record_name} to database")
                    else:
                        print(f"Predictions already exist for record {record_name}, skipping save")
                    
                    # Save physiological metrics if available and don't already exist
                    if stats:
                        existing_metrics = db_manager.get_metrics(record_id)
                        
                        if not existing_metrics:
                            metrics_dict = {}
                            if 'hrv_time' in stats:
                                metrics_dict.update({
                                    'mean_rri_ms': stats['hrv_time'].get('mean_rri_ms'),
                                    'sdnn_ms': stats['hrv_time'].get('sdnn_ms'),
                                    'rmssd_ms': stats['hrv_time'].get('rmssd_ms'),
                                    'pnn50_percent': stats['hrv_time'].get('pnn50_percent'),
                                    'cv_percent': stats['hrv_time'].get('cv_percent')
                                })
                            if 'hrv_freq' in stats:
                                metrics_dict.update({
                                    'vlf_power_ms2': stats['hrv_freq'].get('vlf_power_ms2'),
                                    'lf_power_ms2': stats['hrv_freq'].get('lf_power_ms2'),
                                    'hf_power_ms2': stats['hrv_freq'].get('hf_power_ms2'),
                                    'total_power_ms2': stats['hrv_freq'].get('total_power_ms2'),
                                    'lf_hf_ratio': stats['hrv_freq'].get('lf_hf_ratio')
                                })
                            if 'edr' in stats:
                                metrics_dict.update({
                                    'resp_rate_bpm': stats['edr'].get('resp_rate_bpm'),
                                    'edr_amplitude_range': stats['edr'].get('edr_amplitude_range'),
                                    'edr_variability': stats['edr'].get('edr_variability')
                                })
                            if 'rpeak' in stats:
                                metrics_dict.update({
                                    'num_rpeaks': stats['rpeak'].get('num_rpeaks'),
                                    'mean_hr_bpm': stats['rpeak'].get('mean_hr_bpm'),
                                    'hr_std_bpm': stats['rpeak'].get('hr_std_bpm'),
                                    'recording_duration_min': stats['rpeak'].get('recording_duration_min')
                                })
                            
                            if metrics_dict:
                                # Convert numpy types in metrics_dict before saving to database
                                metrics_dict_clean = convert_numpy_types(metrics_dict)
                                db_manager.save_metrics(record_id, metrics_dict_clean)
                                print(f"Saved metrics for {record_name} to database")
                        else:
                            print(f"Metrics already exist for record {record_name}, skipping save")
                
            except Exception as db_error:
                # Database logging is optional, don't fail the request
                print(f"Warning: Failed to log to database: {db_error}")
        
        # Convert all data to serializable format (predictions, stats, skipped)
        predictions_serializable = convert_numpy_types(predictions)
        stats_serializable = convert_numpy_types(stats)
        skipped_serializable = convert_numpy_types(skipped)
        
        return {
            "filename": filename,
            "predictions": predictions_serializable,
            "skipped": skipped_serializable,
            "stats": stats_serializable
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
        
        # Optional: Log to database if available
        if db_manager:
            try:
                # Get or create record
                record = db_manager.get_record_by_name(record_name)
                if record:
                    # Save Grad-CAM image reference to database
                    db_manager.save_gradcam_image(
                        record_id=record['id'],
                        minute=int(target_minute),
                        image_path=image_path,
                        probability=confidence,
                        predicted_class=class_label
                    )
            except Exception as db_error:
                # Database logging is optional, don't fail the request
                print(f"Warning: Failed to log Grad-CAM to database: {db_error}")
        
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

