import sqlite3
import os

# Store DB in data directory
DB_PATH = os.path.join(os.path.dirname(__file__), 'patientdb.db')

def init_database():
    """Initialize the patient database with improved schema."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    
    # Enable foreign keys
    cur.execute('PRAGMA foreign_keys = ON')
    
    # Patients table - core demographic information
    cur.execute('''CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        date_of_birth DATE,
        gender TEXT CHECK(gender IN ('M', 'F', 'Other', 'Unknown')),
        weight_kg REAL,
        height_cm REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Records table - ECG recordings
    cur.execute('''CREATE TABLE IF NOT EXISTS records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL,
        record_name TEXT UNIQUE NOT NULL,
        record_type TEXT DEFAULT 'ECG',
        recorded_at DATETIME,
        file_path TEXT NOT NULL,
        duration_minutes REAL,
        sample_rate_hz INTEGER DEFAULT 100,
        notes TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
    )''')
    
    # Predictions table - stores prediction results
    cur.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        record_id INTEGER NOT NULL,
        predictions_json TEXT NOT NULL,
        apnea_minutes INTEGER,
        normal_minutes INTEGER,
        processed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (record_id) REFERENCES records(id) ON DELETE CASCADE
    )''')
    
    # Grad-CAM images table - tracks generated explainability images
    cur.execute('''CREATE TABLE IF NOT EXISTS gradcam_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        record_id INTEGER NOT NULL,
        minute INTEGER NOT NULL,
        image_path TEXT NOT NULL,
        probability REAL,
        predicted_class TEXT,
        generated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (record_id) REFERENCES records(id) ON DELETE CASCADE,
        UNIQUE(record_id, minute)
    )''')
    
    # Physiological metrics table - HRV, EDR, and R-peak statistics
    cur.execute('''CREATE TABLE IF NOT EXISTS physiological_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        record_id INTEGER NOT NULL,
        mean_rri_ms REAL,
        sdnn_ms REAL,
        rmssd_ms REAL,
        pnn50_percent REAL,
        cv_percent REAL,
        vlf_power_ms2 REAL,
        lf_power_ms2 REAL,
        hf_power_ms2 REAL,
        total_power_ms2 REAL,
        lf_hf_ratio REAL,
        resp_rate_bpm REAL,
        edr_amplitude_range REAL,
        edr_variability REAL,
        num_rpeaks INTEGER,
        mean_hr_bpm REAL,
        hr_std_bpm REAL,
        recording_duration_min REAL,
        computed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (record_id) REFERENCES records(id) ON DELETE CASCADE
    )''')
    
    # Create indexes for common queries
    cur.execute('CREATE INDEX IF NOT EXISTS idx_records_patient ON records(patient_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_records_name ON records(record_name)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_predictions_record ON predictions(record_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_gradcam_record ON gradcam_images(record_id)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_gradcam_minute ON gradcam_images(record_id, minute)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_metrics_record ON physiological_metrics(record_id)')
    
    con.commit()
    con.close()
    
    return DB_PATH

if __name__ == '__main__':
    db_path = init_database()
    print(f"Database initialized successfully at: {db_path}")
