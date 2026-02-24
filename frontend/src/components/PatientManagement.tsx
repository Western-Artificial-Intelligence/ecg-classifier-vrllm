import React, { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { PatientAPI, RecordAPI, FilesAPI, APIError } from '../services/api';
import type { Patient, PatientCreate, PatientUpdate, Record } from '../types/database';
import styles from '../styles/PatientManagement.module.css';

type ViewMode = 'list' | 'detail';

const PatientManagement: React.FC = () => {
  const navigate = useNavigate();
  const { patientId } = useParams<{ patientId: string }>();
  
  const [viewMode, setViewMode] = useState<ViewMode>('list');
  const [patients, setPatients] = useState<Patient[]>([]);
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [patientRecords, setPatientRecords] = useState<Record[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'name' | 'date'>('date');
  
  // Modal states
  const [showModal, setShowModal] = useState(false);
  const [modalMode, setModalMode] = useState<'create' | 'edit'>('create');
  const [formData, setFormData] = useState<PatientCreate>({ name: '' });
  const [formErrors, setFormErrors] = useState<string[]>([]);
  const [submitting, setSubmitting] = useState(false);
  
  // File linking state
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [availableFiles, setAvailableFiles] = useState<Array<{
    filename: string;
    record_name: string;
    size: number;
    modified: number;
    is_linked: boolean;
  }>>([]);
  const [selectedFile, setSelectedFile] = useState<string>('');
  const [linking, setLinking] = useState(false);
  const [loadingFiles, setLoadingFiles] = useState(false);

  // Load patients on mount
  useEffect(() => {
    loadPatients();
  }, []);

  // Load patient detail if patientId is in URL
  useEffect(() => {
    if (patientId) {
      const id = parseInt(patientId, 10);
      if (!isNaN(id)) {
        loadPatientDetail(id);
      }
    } else {
      setViewMode('list');
      setSelectedPatient(null);
      setPatientRecords([]);
    }
  }, [patientId]);

  const loadPatients = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await PatientAPI.getAll();
      setPatients(data);
    } catch (err) {
      if (err instanceof APIError && err.statusCode === 503) {
        setError('Database not available. Please check backend connection.');
      } else {
        setError(err instanceof Error ? err.message : 'Failed to load patients');
      }
    } finally {
      setLoading(false);
    }
  };

  const loadPatientDetail = async (id: number) => {
    setLoading(true);
    setError(null);
    try {
      const patient = await PatientAPI.getById(id);
      const records = await PatientAPI.getRecords(id);
      setSelectedPatient(patient);
      setPatientRecords(records);
      setViewMode('detail');
    } catch (err) {
      if (err instanceof APIError && err.statusCode === 404) {
        setError('Patient not found');
        navigate('/patient-management');
      } else {
        setError(err instanceof Error ? err.message : 'Failed to load patient details');
      }
    } finally {
      setLoading(false);
    }
  };

  const handlePatientClick = (patient: Patient) => {
    navigate(`/patient-management/${patient.id}`);
  };

  const handleRecordClick = (record: Record) => {
    // Navigate to analysis view with patient and record context
    navigate('/analysis', {
      state: {
        patient: selectedPatient,
        record: record,
        filename: `${record.record_name}.dat`,
      },
    });
  };

  const handleBackToList = () => {
    navigate('/patient-management');
  };

  const handleAddPatient = () => {
    setModalMode('create');
    setFormData({ name: '' });
    setFormErrors([]);
    setShowModal(true);
  };

  const handleEditPatient = () => {
    if (!selectedPatient) return;
    setModalMode('edit');
    setFormData({
      name: selectedPatient.name,
      date_of_birth: selectedPatient.date_of_birth || undefined,
      gender: selectedPatient.gender || undefined,
      weight_kg: selectedPatient.weight_kg || undefined,
      height_cm: selectedPatient.height_cm || undefined,
    });
    setFormErrors([]);
    setShowModal(true);
  };

  const handleAddRecording = async () => {
    setShowUploadModal(true);
    setSelectedFile('');
    setFormErrors([]);
    
    // Load available files
    setLoadingFiles(true);
    try {
      const result = await FilesAPI.getAvailable();
      setAvailableFiles(result.files);
    } catch (err) {
      setFormErrors([err instanceof Error ? err.message : 'Failed to load available files']);
    } finally {
      setLoadingFiles(false);
    }
  };

  const handleLinkSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!selectedFile || !selectedPatient) {
      setFormErrors(['Please select a file']);
      return;
    }

    setLinking(true);
    setFormErrors([]);

    try {
      const fileInfo = availableFiles.find(f => f.filename === selectedFile);
      if (!fileInfo) {
        setFormErrors(['File not found']);
        return;
      }

      // Create record linking patient to file
      // Backend will construct the full path using config.RAW_DATA_DIR
      await RecordAPI.create({
        patient_id: selectedPatient.id,
        record_name: fileInfo.record_name,
        file_path: fileInfo.filename,
        sample_rate_hz: 100
      });

      // Reload patient records
      const records = await PatientAPI.getRecords(selectedPatient.id);
      setPatientRecords(records);
      
      // Close modal
      setShowUploadModal(false);
      setSelectedFile('');
    } catch (err) {
      if (err instanceof APIError && err.statusCode === 400) {
        setFormErrors(['This file is already linked to a patient']);
      } else {
        setFormErrors([err instanceof Error ? err.message : 'Failed to link file']);
      }
    } finally {
      setLinking(false);
    }
  };

  const handleCloseModal = () => {
    setShowModal(false);
    setFormData({ name: '' });
    setFormErrors([]);
  };

  const validateForm = (): boolean => {
    const errors: string[] = [];
    
    if (!formData.name || formData.name.trim() === '') {
      errors.push('Name is required');
    }
    
    if (formData.weight_kg && formData.weight_kg <= 0) {
      errors.push('Weight must be positive');
    }
    
    if (formData.height_cm && formData.height_cm <= 0) {
      errors.push('Height must be positive');
    }
    
    setFormErrors(errors);
    return errors.length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) return;
    
    setSubmitting(true);
    try {
      if (modalMode === 'create') {
        await PatientAPI.create(formData);
        await loadPatients();
      } else if (selectedPatient) {
        const updates: PatientUpdate = {};
        if (formData.name) updates.name = formData.name;
        if (formData.date_of_birth) updates.date_of_birth = formData.date_of_birth;
        if (formData.gender) updates.gender = formData.gender;
        if (formData.weight_kg) updates.weight_kg = formData.weight_kg;
        if (formData.height_cm) updates.height_cm = formData.height_cm;
        
        const updated = await PatientAPI.update(selectedPatient.id, updates);
        setSelectedPatient(updated);
        await loadPatients();
      }
      handleCloseModal();
    } catch (err) {
      setFormErrors([err instanceof Error ? err.message : 'Failed to save patient']);
    } finally {
      setSubmitting(false);
    }
  };

  // Filter and sort patients
  const filteredPatients = patients
    .filter((p) => p.name.toLowerCase().includes(searchQuery.toLowerCase()))
    .sort((a, b) => {
      if (sortBy === 'name') {
        return a.name.localeCompare(b.name);
      } else {
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      }
    });

  if (loading && patients.length === 0) {
    return (
      <div className={styles.container}>
        <div className={styles.loading}>Loading patients...</div>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.headerContent}>
          <button className={styles.backButton} onClick={() => navigate('/')}>
            ← Back to Landing Page
          </button>
          <h1 className={styles.title}>
            {viewMode === 'list' ? 'Patient Management' : selectedPatient?.name}
          </h1>
        </div>
      </header>

      {error && (
        <div className={styles.error}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {viewMode === 'list' ? (
        <div className={styles.listView}>
          <div className={styles.controls}>
            <input
              type="text"
              className={styles.searchInput}
              placeholder="Search patients..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <select
              className={styles.sortSelect}
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as 'name' | 'date')}
            >
              <option value="date">Sort by Date</option>
              <option value="name">Sort by Name</option>
            </select>
            <button className={styles.addButton} onClick={handleAddPatient}>
              + Add New Patient
            </button>
          </div>

          {filteredPatients.length === 0 ? (
            <div className={styles.emptyState}>
              <p>No patients found.</p>
              {patients.length === 0 && (
                <button className={styles.primaryButton} onClick={handleAddPatient}>
                  Create First Patient
                </button>
              )}
            </div>
          ) : (
            <div className={styles.patientGrid}>
              {filteredPatients.map((patient) => (
                <div
                  key={patient.id}
                  className={styles.patientCard}
                  onClick={() => handlePatientClick(patient)}
                >
                  <div className={styles.patientCardHeader}>
                    <h3>{patient.name}</h3>
                    <span className={styles.patientId}>ID: {patient.id}</span>
                  </div>
                  <div className={styles.patientCardBody}>
                    {patient.date_of_birth && (
                      <div className={styles.patientInfo}>
                        <strong>DOB:</strong> {patient.date_of_birth}
                      </div>
                    )}
                    {patient.gender && (
                      <div className={styles.patientInfo}>
                        <strong>Gender:</strong> {patient.gender}
                      </div>
                    )}
                    <div className={styles.patientInfo}>
                      <strong>Created:</strong> {new Date(patient.created_at).toLocaleDateString()}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div className={styles.detailView}>
          <div className={styles.detailHeader}>
            <button className={styles.backButton} onClick={handleBackToList}>
              ← Back to List
            </button>
            <button className={styles.editButton} onClick={handleEditPatient}>
              Edit Patient
            </button>
          </div>

          {selectedPatient && (
            <div className={styles.patientDetails}>
              <h2>Patient Information</h2>
              <div className={styles.detailGrid}>
                <div className={styles.detailItem}>
                  <strong>Name:</strong> {selectedPatient.name}
                </div>
                <div className={styles.detailItem}>
                  <strong>ID:</strong> {selectedPatient.id}
                </div>
                {selectedPatient.date_of_birth && (
                  <div className={styles.detailItem}>
                    <strong>Date of Birth:</strong> {selectedPatient.date_of_birth}
                  </div>
                )}
                {selectedPatient.gender && (
                  <div className={styles.detailItem}>
                    <strong>Gender:</strong> {selectedPatient.gender}
                  </div>
                )}
                {selectedPatient.weight_kg && (
                  <div className={styles.detailItem}>
                    <strong>Weight:</strong> {selectedPatient.weight_kg} kg
                  </div>
                )}
                {selectedPatient.height_cm && (
                  <div className={styles.detailItem}>
                    <strong>Height:</strong> {selectedPatient.height_cm} cm
                  </div>
                )}
                <div className={styles.detailItem}>
                  <strong>Created:</strong> {new Date(selectedPatient.created_at).toLocaleString()}
                </div>
              </div>
            </div>
          )}

          <div className={styles.recordsSection}>
            <div className={styles.recordsHeader}>
              <h2>ECG Recordings</h2>
              <button className={styles.addRecordingButton} onClick={handleAddRecording}>
                + Add Recording
              </button>
            </div>
            {patientRecords.length === 0 ? (
              <div className={styles.emptyState}>
                <p>No recordings found for this patient.</p>
                <button className={styles.primaryButton} onClick={handleAddRecording}>
                  Add First Recording
                </button>
              </div>
            ) : (
              <div className={styles.recordsList}>
                {patientRecords.map((record) => (
                  <div
                    key={record.id}
                    className={styles.recordCard}
                    onClick={() => handleRecordClick(record)}
                  >
                    <div className={styles.recordCardHeader}>
                      <h3>{record.record_name}</h3>
                      <span className={styles.recordType}>{record.record_type}</span>
                    </div>
                    <div className={styles.recordCardBody}>
                      {record.duration_minutes && (
                        <div className={styles.recordInfo}>
                          <strong>Duration:</strong> {record.duration_minutes} minutes
                        </div>
                      )}
                      <div className={styles.recordInfo}>
                        <strong>Sample Rate:</strong> {record.sample_rate_hz} Hz
                      </div>
                      {record.recorded_at && (
                        <div className={styles.recordInfo}>
                          <strong>Recorded:</strong> {new Date(record.recorded_at).toLocaleDateString()}
                        </div>
                      )}
                      <div className={styles.recordInfo}>
                        <strong>Added:</strong> {new Date(record.created_at).toLocaleDateString()}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {showModal && (
        <div className={styles.modal} onClick={handleCloseModal}>
          <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h2>{modalMode === 'create' ? 'Add New Patient' : 'Edit Patient'}</h2>
              <button className={styles.modalClose} onClick={handleCloseModal}>
                ×
              </button>
            </div>
            <form onSubmit={handleSubmit} className={styles.modalForm}>
              {formErrors.length > 0 && (
                <div className={styles.formErrors}>
                  {formErrors.map((err, idx) => (
                    <div key={idx}>{err}</div>
                  ))}
                </div>
              )}
              
              <div className={styles.formGroup}>
                <label htmlFor="name">Name *</label>
                <input
                  id="name"
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  required
                />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="dob">Date of Birth</label>
                <input
                  id="dob"
                  type="date"
                  value={formData.date_of_birth || ''}
                  onChange={(e) => setFormData({ ...formData, date_of_birth: e.target.value })}
                />
              </div>

              <div className={styles.formGroup}>
                <label htmlFor="gender">Gender</label>
                <select
                  id="gender"
                  value={formData.gender || ''}
                  onChange={(e) => setFormData({ ...formData, gender: e.target.value as any })}
                >
                  <option value="">Select...</option>
                  <option value="M">Male</option>
                  <option value="F">Female</option>
                  <option value="Other">Other</option>
                  <option value="Unknown">Unknown</option>
                </select>
              </div>

              <div className={styles.formRow}>
                <div className={styles.formGroup}>
                  <label htmlFor="weight">Weight (kg)</label>
                  <input
                    id="weight"
                    type="number"
                    step="0.1"
                    value={formData.weight_kg || ''}
                    onChange={(e) => setFormData({ ...formData, weight_kg: parseFloat(e.target.value) || undefined })}
                  />
                </div>

                <div className={styles.formGroup}>
                  <label htmlFor="height">Height (cm)</label>
                  <input
                    id="height"
                    type="number"
                    step="0.1"
                    value={formData.height_cm || ''}
                    onChange={(e) => setFormData({ ...formData, height_cm: parseFloat(e.target.value) || undefined })}
                  />
                </div>
              </div>

              <div className={styles.modalActions}>
                <button type="button" className={styles.cancelButton} onClick={handleCloseModal}>
                  Cancel
                </button>
                <button type="submit" className={styles.submitButton} disabled={submitting}>
                  {submitting ? 'Saving...' : modalMode === 'create' ? 'Create Patient' : 'Save Changes'}
                </button>
          </div>
        </form>
      </div>
    </div>
  )}

      {showUploadModal && (
        <div className={styles.modal} onClick={() => setShowUploadModal(false)}>
          <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h2>Add ECG Recording</h2>
              <button className={styles.modalClose} onClick={() => setShowUploadModal(false)}>
                ×
              </button>
            </div>
            <form onSubmit={handleLinkSubmit} className={styles.modalForm}>
              {formErrors.length > 0 && (
                <div className={styles.formErrors}>
                  {formErrors.map((err, idx) => (
                    <div key={idx}>{err}</div>
                  ))}
                </div>
              )}
              
              <div className={styles.formGroup}>
                <label htmlFor="file-select">Select .dat file from data directory</label>
                {loadingFiles ? (
                  <div className={styles.loadingFiles}>Loading available files...</div>
                ) : (
                  <select
                    id="file-select"
                    value={selectedFile}
                    onChange={(e) => setSelectedFile(e.target.value)}
                    required
                  >
                    <option value="">Choose a file...</option>
                    {availableFiles.map((file) => (
                      <option 
                        key={file.filename} 
                        value={file.filename}
                        disabled={file.is_linked}
                      >
                        {file.filename} 
                        {file.is_linked ? ' (already linked)' : ''} 
                        - {(file.size / 1024 / 1024).toFixed(2)} MB
                      </option>
                    ))}
                  </select>
                )}
              </div>

              {availableFiles.length === 0 && !loadingFiles && (
                <div className={styles.uploadNote}>
                  <strong>Note:</strong> No .dat files found in the data/raw directory.
                  Please add ECG files to the backend data/raw directory.
                </div>
              )}

              <div className={styles.modalActions}>
                <button
                  type="button"
                  className={styles.cancelButton}
                  onClick={() => setShowUploadModal(false)}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className={styles.submitButton}
                  disabled={!selectedFile || linking || loadingFiles}
                >
                  {linking ? 'Linking...' : 'Link Recording'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default PatientManagement;
