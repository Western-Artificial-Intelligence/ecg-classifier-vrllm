import os
import struct
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# Define the base directory for ECG data relative to the project root
# This assumes the backend is run from the project root or similar.
ECG_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/raw/ecgdata")

app = FastAPI()

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

@app.get("/api/status")
async def get_status():
    """Simple endpoint to check if the API is running."""
    return {"status": "ok", "message": "ECG Backend API is running."}

