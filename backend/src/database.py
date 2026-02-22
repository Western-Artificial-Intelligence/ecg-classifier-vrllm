"""
Database manager for ECG patient records and analysis results.

This module provides a high-level interface for interacting with the SQLite database,
managing patients, ECG recordings, predictions, Grad-CAM images, and physiological metrics.
"""

import sqlite3
import os
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

# Path to the database file
DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'patientdb.db')


class DatabaseManager:
    """Manages all database operations for the ECG triage system."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the database manager.
        
        Args:
            db_path: Optional custom path to the database file. 
                    If None, uses the default path in data/patientdb.db
        """
        self.db_path = db_path or DB_PATH
        self._ensure_initialized()
    
    def _ensure_initialized(self):
        """Ensure the database is initialized with the correct schema."""
        if not os.path.exists(self.db_path):
            # Initialize the database if it doesn't exist
            from data.patientdb import init_database
            init_database()
    
    def get_connection(self) -> sqlite3.Connection:
        """
        Get a database connection with row factory enabled.
        
        Returns:
            A SQLite connection object configured to return dict-like rows.
        """
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row  # Return dict-like rows
        con.execute('PRAGMA foreign_keys = ON')  # Enable foreign key constraints
        return con
    
    # ========== Patient Operations ==========
    
    def add_patient(self, name: str, date_of_birth: Optional[str] = None,
                   gender: Optional[str] = None, weight_kg: Optional[float] = None,
                   height_cm: Optional[float] = None) -> int:
        """
        Add a new patient to the database.
        
        Args:
            name: Patient's name (required)
            date_of_birth: Date of birth in YYYY-MM-DD format
            gender: Gender ('M', 'F', 'Other', 'Unknown')
            weight_kg: Weight in kilograms
            height_cm: Height in centimeters
            
        Returns:
            The ID of the newly created patient record
        """
        with self.get_connection() as con:
            cur = con.cursor()
            cur.execute('''INSERT INTO patients 
                (name, date_of_birth, gender, weight_kg, height_cm)
                VALUES (?, ?, ?, ?, ?)''',
                (name, date_of_birth, gender, weight_kg, height_cm))
            con.commit()
            return cur.lastrowid
    
    def get_patient(self, patient_id: int) -> Optional[Dict[str, Any]]:
        """
        Get patient information by ID.
        
        Args:
            patient_id: The patient's ID
            
        Returns:
            Dictionary containing patient information, or None if not found
        """
        with self.get_connection() as con:
            cur = con.execute('SELECT * FROM patients WHERE id = ?', (patient_id,))
            row = cur.fetchone()
            return dict(row) if row else None
    
    def get_all_patients(self) -> List[Dict[str, Any]]:
        """
        Get all patients in the database.
        
        Returns:
            List of dictionaries containing patient information
        """
        with self.get_connection() as con:
            cur = con.execute('SELECT * FROM patients ORDER BY created_at DESC')
            return [dict(row) for row in cur.fetchall()]
    
    def update_patient(self, patient_id: int, **kwargs) -> bool:
        """
        Update patient information.
        
        Args:
            patient_id: The patient's ID
            **kwargs: Fields to update (name, date_of_birth, gender, weight_kg, height_cm)
            
        Returns:
            True if update was successful, False otherwise
        """
        allowed_fields = {'name', 'date_of_birth', 'gender', 'weight_kg', 'height_cm'}
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        
        if not updates:
            return False
        
        set_clause = ', '.join(f"{field} = ?" for field in updates.keys())
        query = f"UPDATE patients SET {set_clause} WHERE id = ?"
        
        with self.get_connection() as con:
            con.execute(query, list(updates.values()) + [patient_id])
            con.commit()
            return True
    
    # ========== Record Operations ==========
    
    def add_record(self, patient_id: int, record_name: str, file_path: str,
                  record_type: str = 'ECG', recorded_at: Optional[str] = None,
                  duration_minutes: Optional[float] = None, sample_rate_hz: int = 100,
                  notes: Optional[str] = None) -> int:
        """
        Add a new ECG recording.
        
        Args:
            patient_id: ID of the patient this record belongs to
            record_name: Unique identifier for the recording (e.g., 'a01')
            file_path: Path to the .dat file
            record_type: Type of recording (default: 'ECG')
            recorded_at: Timestamp when recorded (YYYY-MM-DD HH:MM:SS)
            duration_minutes: Length of recording in minutes
            sample_rate_hz: Sampling rate in Hz (default: 100)
            notes: Optional notes about the recording
            
        Returns:
            The ID of the newly created record
        """
        with self.get_connection() as con:
            cur = con.cursor()
            cur.execute('''INSERT INTO records 
                (patient_id, record_name, file_path, record_type, recorded_at,
                 duration_minutes, sample_rate_hz, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (patient_id, record_name, file_path, record_type, recorded_at,
                 duration_minutes, sample_rate_hz, notes))
            con.commit()
            return cur.lastrowid
    
    def get_record(self, record_id: int) -> Optional[Dict[str, Any]]:
        """
        Get record information by ID.
        
        Args:
            record_id: The record's ID
            
        Returns:
            Dictionary containing record information, or None if not found
        """
        with self.get_connection() as con:
            cur = con.execute('SELECT * FROM records WHERE id = ?', (record_id,))
            row = cur.fetchone()
            return dict(row) if row else None
    
    def get_record_by_name(self, record_name: str) -> Optional[Dict[str, Any]]:
        """
        Get record information by record name.
        
        Args:
            record_name: The record's unique name (e.g., 'a01')
            
        Returns:
            Dictionary containing record information, or None if not found
        """
        with self.get_connection() as con:
            cur = con.execute('SELECT * FROM records WHERE record_name = ?', (record_name,))
            row = cur.fetchone()
            return dict(row) if row else None
    
    def get_patient_records(self, patient_id: int) -> List[Dict[str, Any]]:
        """
        Get all records for a specific patient.
        
        Args:
            patient_id: The patient's ID
            
        Returns:
            List of dictionaries containing record information
        """
        with self.get_connection() as con:
            cur = con.execute(
                'SELECT * FROM records WHERE patient_id = ? ORDER BY created_at DESC',
                (patient_id,))
            return [dict(row) for row in cur.fetchall()]
    
    # ========== Prediction Operations ==========
    
    def save_predictions(self, record_id: int, predictions: List[Dict[str, Any]]) -> int:
        """
        Save prediction results for a recording.
        
        Args:
            record_id: The record ID these predictions belong to
            predictions: List of prediction dictionaries (minute, probability)
            
        Returns:
            The ID of the newly created prediction record
        """
        apnea_count = sum(1 for p in predictions if p.get('probability', 0) >= 0.5)
        normal_count = len(predictions) - apnea_count
        
        with self.get_connection() as con:
            cur = con.cursor()
            cur.execute('''INSERT INTO predictions 
                (record_id, predictions_json, apnea_minutes, normal_minutes)
                VALUES (?, ?, ?, ?)''',
                (record_id, json.dumps(predictions), apnea_count, normal_count))
            con.commit()
            return cur.lastrowid
    
    def get_predictions(self, record_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the most recent predictions for a record.
        
        Args:
            record_id: The record's ID
            
        Returns:
            Dictionary containing prediction information including parsed JSON
        """
        with self.get_connection() as con:
            cur = con.execute(
                '''SELECT * FROM predictions 
                   WHERE record_id = ? 
                   ORDER BY processed_at DESC 
                   LIMIT 1''',
                (record_id,))
            row = cur.fetchone()
            if row:
                result = dict(row)
                result['predictions'] = json.loads(result['predictions_json'])
                return result
            return None
    
    # ========== Grad-CAM Operations ==========
    
    def save_gradcam_image(self, record_id: int, minute: int, image_path: str,
                          probability: Optional[float] = None,
                          predicted_class: Optional[str] = None) -> int:
        """
        Save a Grad-CAM image record.
        
        Args:
            record_id: The record ID this image belongs to
            minute: The minute within the recording
            image_path: Path to the saved image file
            probability: Prediction probability for this minute
            predicted_class: Predicted class (e.g., 'Apnea', 'Non-Apnea')
            
        Returns:
            The ID of the newly created image record
        """
        with self.get_connection() as con:
            cur = con.cursor()
            cur.execute('''INSERT OR REPLACE INTO gradcam_images 
                (record_id, minute, image_path, probability, predicted_class)
                VALUES (?, ?, ?, ?, ?)''',
                (record_id, minute, image_path, probability, predicted_class))
            con.commit()
            return cur.lastrowid
    
    def get_gradcam_images(self, record_id: int) -> List[Dict[str, Any]]:
        """
        Get all Grad-CAM images for a record.
        
        Args:
            record_id: The record's ID
            
        Returns:
            List of dictionaries containing image information
        """
        with self.get_connection() as con:
            cur = con.execute(
                '''SELECT * FROM gradcam_images 
                   WHERE record_id = ? 
                   ORDER BY minute ASC''',
                (record_id,))
            return [dict(row) for row in cur.fetchall()]
    
    def get_gradcam_image(self, record_id: int, minute: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific Grad-CAM image.
        
        Args:
            record_id: The record's ID
            minute: The minute within the recording
            
        Returns:
            Dictionary containing image information, or None if not found
        """
        with self.get_connection() as con:
            cur = con.execute(
                'SELECT * FROM gradcam_images WHERE record_id = ? AND minute = ?',
                (record_id, minute))
            row = cur.fetchone()
            return dict(row) if row else None
    
    # ========== Physiological Metrics Operations ==========
    
    def save_metrics(self, record_id: int, metrics: Dict[str, Any]) -> int:
        """
        Save physiological metrics for a recording.
        
        Args:
            record_id: The record ID these metrics belong to
            metrics: Dictionary containing metric values
            
        Returns:
            The ID of the newly created metrics record
        """
        with self.get_connection() as con:
            cur = con.cursor()
            cur.execute('''INSERT INTO physiological_metrics 
                (record_id, mean_rri_ms, sdnn_ms, rmssd_ms, pnn50_percent, cv_percent,
                 vlf_power_ms2, lf_power_ms2, hf_power_ms2, total_power_ms2, lf_hf_ratio,
                 resp_rate_bpm, edr_amplitude_range, edr_variability,
                 num_rpeaks, mean_hr_bpm, hr_std_bpm, recording_duration_min)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (record_id,
                 metrics.get('mean_rri_ms'), metrics.get('sdnn_ms'), metrics.get('rmssd_ms'),
                 metrics.get('pnn50_percent'), metrics.get('cv_percent'),
                 metrics.get('vlf_power_ms2'), metrics.get('lf_power_ms2'),
                 metrics.get('hf_power_ms2'), metrics.get('total_power_ms2'),
                 metrics.get('lf_hf_ratio'), metrics.get('resp_rate_bpm'),
                 metrics.get('edr_amplitude_range'), metrics.get('edr_variability'),
                 metrics.get('num_rpeaks'), metrics.get('mean_hr_bpm'),
                 metrics.get('hr_std_bpm'), metrics.get('recording_duration_min')))
            con.commit()
            return cur.lastrowid
    
    def get_metrics(self, record_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the most recent physiological metrics for a record.
        
        Args:
            record_id: The record's ID
            
        Returns:
            Dictionary containing metric values, or None if not found
        """
        with self.get_connection() as con:
            cur = con.execute(
                '''SELECT * FROM physiological_metrics 
                   WHERE record_id = ? 
                   ORDER BY computed_at DESC 
                   LIMIT 1''',
                (record_id,))
            row = cur.fetchone()
            return dict(row) if row else None


# Singleton instance for easy access
_db_manager = None

def get_db_manager() -> DatabaseManager:
    """
    Get the singleton DatabaseManager instance.
    
    Returns:
        The global DatabaseManager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
