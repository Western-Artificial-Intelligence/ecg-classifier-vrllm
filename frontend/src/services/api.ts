/**
 * API Service Layer
 * Centralizes all API calls with typed interfaces and error handling.
 */

import type {
  Patient,
  PatientCreate,
  PatientUpdate,
  PatientsResponse,
  Record,
  RecordsResponse,
  PredictionRecord,
  PhysiologicalMetrics,
  GradcamImage,
  GradcamsResponse,
} from '../types/database';

// Note: ApiError type is defined in database.ts but used here for type checking only
type ApiError = {
  detail: string;
};
  
  const API_BASE_URL = 'http://localhost:8000';
  
  /**
   * Custom error class for API errors
   */
  export class APIError extends Error {
    statusCode: number;

    constructor(statusCode: number, message: string) {
      super(message);
      this.name = 'APIError';
      this.statusCode = statusCode;
    }
  }
  
  /**
   * Helper function to handle API responses
   */
  async function handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
      const error: ApiError = await response.json().catch(() => ({
        detail: `HTTP ${response.status}: ${response.statusText}`,
      }));
      throw new APIError(response.status, error.detail);
    }
    return response.json();
  }
  
  /**
   * Patient API - handles all patient-related operations
   */
  export class PatientAPI {
    /**
     * Get all patients from the database
     */
    static async getAll(): Promise<Patient[]> {
      const response = await fetch(`${API_BASE_URL}/api/patients`);
      const data = await handleResponse<PatientsResponse>(response);
      return data.patients;
    }
  
    /**
     * Get a specific patient by ID
     */
    static async getById(patientId: number): Promise<Patient> {
      const response = await fetch(`${API_BASE_URL}/api/patients/${patientId}`);
      return handleResponse<Patient>(response);
    }
  
    /**
     * Create a new patient
     */
    static async create(patient: PatientCreate): Promise<Patient> {
      const response = await fetch(`${API_BASE_URL}/api/patients`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(patient),
      });
      return handleResponse<Patient>(response);
    }
  
    /**
     * Update an existing patient
     */
    static async update(patientId: number, updates: PatientUpdate): Promise<Patient> {
      const response = await fetch(`${API_BASE_URL}/api/patients/${patientId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updates),
      });
      return handleResponse<Patient>(response);
    }
  
    /**
     * Get all ECG recordings for a specific patient
     */
    static async getRecords(patientId: number): Promise<Record[]> {
      const response = await fetch(`${API_BASE_URL}/api/patients/${patientId}/records`);
      const data = await handleResponse<RecordsResponse>(response);
      return data.records;
    }
  }
  
/**
 * Record API - handles all ECG record-related operations
 */
export class RecordAPI {
  /**
   * Get a specific record by ID
   */
  static async getById(recordId: number): Promise<Record> {
    const response = await fetch(`${API_BASE_URL}/api/records/${recordId}`);
    return handleResponse<Record>(response);
  }

  /**
   * Create a new record linking a patient to a file
   */
  static async create(recordData: {
    patient_id: number;
    record_name: string;
    file_path: string;
    duration_minutes?: number;
    sample_rate_hz?: number;
    recorded_at?: string;
    notes?: string;
  }): Promise<Record> {
    const response = await fetch(`${API_BASE_URL}/api/records`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(recordData),
    });
    return handleResponse<Record>(response);
  }
  
    /**
     * Get predictions for a specific record
     */
    static async getPredictions(recordId: number): Promise<PredictionRecord | null> {
      const response = await fetch(`${API_BASE_URL}/api/records/${recordId}/predictions`);
      const data = await handleResponse<PredictionRecord | { predictions: null; message: string }>(response);
      
      if ('predictions' in data && data.predictions === null) {
        return null;
      }
      
      const predRecord = data as PredictionRecord;
      // Parse predictions_json if it's a string
      if (predRecord.predictions_json && typeof predRecord.predictions_json === 'string') {
        predRecord.predictions = JSON.parse(predRecord.predictions_json);
      }
      
      return predRecord;
    }
  
    /**
     * Get physiological metrics for a specific record
     */
    static async getMetrics(recordId: number): Promise<PhysiologicalMetrics | null> {
      const response = await fetch(`${API_BASE_URL}/api/records/${recordId}/metrics`);
      const data = await handleResponse<PhysiologicalMetrics | { metrics: null; message: string }>(response);
      
      if ('metrics' in data && data.metrics === null) {
        return null;
      }
      
      return data as PhysiologicalMetrics;
    }
  
    /**
     * Get all Grad-CAM images metadata for a specific record
     */
    static async getGradcams(recordId: number): Promise<GradcamImage[]> {
      const response = await fetch(`${API_BASE_URL}/api/records/${recordId}/gradcams`);
      const data = await handleResponse<GradcamsResponse>(response);
      return data.gradcams;
    }
  }
  
  /**
   * Analysis API - handles prediction and Grad-CAM generation
   */
  export class AnalysisAPI {
    /**
     * Get ECG data for a file
     */
    static async getEcgData(filename: string): Promise<{ filename: string; data: number[] }> {
      const response = await fetch(`${API_BASE_URL}/api/ecg_data/${filename}`);
      return handleResponse(response);
    }
  
    /**
     * Run prediction on an ECG record
     */
    static async predict(filename: string): Promise<{
      filename: string;
      predictions: { minute: number; probability: number }[];
      skipped: number[];
      stats?: any;
    }> {
      const response = await fetch(`${API_BASE_URL}/api/predict/${filename}`, {
        method: 'POST',
      });
      return handleResponse(response);
    }
  
    /**
     * Generate Grad-CAM visualization for a specific minute
     */
    static async generateGradcam(
      filename: string,
      minute?: number
    ): Promise<{
      filename: string;
      minute: number;
      image_url: string;
      probability: number;
      predicted_class: string;
    }> {
      const url = minute !== undefined
        ? `${API_BASE_URL}/api/gradcam/${filename}?minute=${minute}`
        : `${API_BASE_URL}/api/gradcam/${filename}`;
      
      const response = await fetch(url, { method: 'POST' });
      return handleResponse(response);
    }
  
    /**
     * Generate Grad-CAM visualizations for top N predictions
     */
    static async generateGradcamBatch(
      filename: string,
      count: number = 3,
      signal?: AbortSignal
    ): Promise<{
      filename: string;
      results: Array<{
        minute: number;
        image_url: string;
        probability: number;
        predicted_class: string;
      }>;
    }> {
      const response = await fetch(
        `${API_BASE_URL}/api/gradcam/batch/${filename}?count=${count}`,
        {
          method: 'POST',
          signal,
        }
      );
      return handleResponse(response);
    }
  
    /**
     * Get Grad-CAM heatmap data for a minute (for overlay on ECG chart).
     * Returns heatmap array of length 6000 (100 Hz × 60 s), values in [0, 1].
     */
    static async getGradcamHeatmap(
      filename: string,
      minute: number
    ): Promise<{
      minute: number;
      heatmap: number[];
      probability: number;
      predicted_class: string;
    }> {
      const response = await fetch(
        `${API_BASE_URL}/api/gradcam/${filename}/heatmap?minute=${minute}`
      );
      return handleResponse(response);
    }

    /**
     * List all available Grad-CAM images for a record
     */
    static async listGradcams(filename: string): Promise<{
      filename: string;
      images: Array<{
        minute: number;
        image_url: string;
        probability?: number;
        predicted_class?: string;
      }>;
    }> {
      const response = await fetch(`${API_BASE_URL}/api/gradcam/list/${filename}`);
      return handleResponse(response);
    }
  
    /**
     * Generate AI agent analysis for a record
     */
    static async generateAgentAnalysis(filename: string): Promise<{
      filename: string;
      analysis: string;
      source: string;
      meta: { analyzed_minutes: number };
    }> {
      const response = await fetch(`${API_BASE_URL}/api/agent/analyze/${filename}`, {
        method: 'POST',
      });
      return handleResponse(response);
    }
  }
  
/**
 * Files API - handles file management operations
 */
export class FilesAPI {
  /**
   * Get list of available .dat files in the raw data directory
   */
  static async getAvailable(): Promise<{
    files: Array<{
      filename: string;
      record_name: string;
      size: number;
      modified: number;
      is_linked: boolean;
    }>;
  }> {
    const response = await fetch(`${API_BASE_URL}/api/files/available`);
    return handleResponse(response);
  }
}

/**
 * Helper function to check if database is available
 */
export async function isDatabaseAvailable(): Promise<boolean> {
  try {
    await PatientAPI.getAll();
    return true;
  } catch (error) {
    if (error instanceof APIError && error.statusCode === 503) {
      return false;
    }
    throw error;
  }
}
  
