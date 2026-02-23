/**
 * TypeScript interfaces matching the backend database schema.
 * These types are used throughout the application for type safety.
 */

export interface Patient {
    id: number;
    name: string;
    date_of_birth?: string | null;
    gender?: 'M' | 'F' | 'Other' | 'Unknown' | null;
    weight_kg?: number | null;
    height_cm?: number | null;
    created_at: string;
  }
  
  export interface PatientCreate {
    name: string;
    date_of_birth?: string;
    gender?: 'M' | 'F' | 'Other' | 'Unknown';
    weight_kg?: number;
    height_cm?: number;
  }
  
  export interface PatientUpdate {
    name?: string;
    date_of_birth?: string;
    gender?: 'M' | 'F' | 'Other' | 'Unknown';
    weight_kg?: number;
    height_cm?: number;
  }
  
  export interface Record {
    id: number;
    patient_id: number;
    record_name: string;
    record_type: string;
    recorded_at?: string | null;
    file_path: string;
    duration_minutes?: number | null;
    sample_rate_hz: number;
    notes?: string | null;
    created_at: string;
  }
  
  export interface Prediction {
    minute: number;
    probability: number;
  }
  
  export interface PredictionRecord {
    id: number;
    record_id: number;
    predictions_json: string;
    apnea_minutes: number;
    normal_minutes: number;
    processed_at: string;
    predictions?: Prediction[];
  }
  
  export interface GradcamImage {
    id: number;
    record_id: number;
    minute: number;
    image_path: string;
    probability?: number | null;
    predicted_class?: string | null;
    generated_at: string;
  }
  
  export interface PhysiologicalMetrics {
    id: number;
    record_id: number;
    // HRV Time Domain
    mean_rri_ms?: number | null;
    sdnn_ms?: number | null;
    rmssd_ms?: number | null;
    pnn50_percent?: number | null;
    cv_percent?: number | null;
    // HRV Frequency Domain
    vlf_power_ms2?: number | null;
    lf_power_ms2?: number | null;
    hf_power_ms2?: number | null;
    total_power_ms2?: number | null;
    lf_hf_ratio?: number | null;
    // EDR (ECG-Derived Respiration)
    resp_rate_bpm?: number | null;
    edr_amplitude_range?: number | null;
    edr_variability?: number | null;
    // R-Peak Statistics
    num_rpeaks?: number | null;
    mean_hr_bpm?: number | null;
    hr_std_bpm?: number | null;
    recording_duration_min?: number | null;
    computed_at: string;
  }
  
  // API Response types
  export interface PatientsResponse {
    patients: Patient[];
  }
  
  export interface RecordsResponse {
    records: Record[];
  }
  
  export interface GradcamsResponse {
    gradcams: GradcamImage[];
  }
  
  export interface ApiError {
    detail: string;
  }
  