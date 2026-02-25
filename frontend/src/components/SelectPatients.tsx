import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { PatientAPI, APIError } from '../services/api';
import type { Patient } from '../types/database';
import styles from '../styles/SelectPatients.module.css';


interface SelectedPatient {
  id: string;
  name: string;
  displayId: string;
}

const PLACEHOLDER_PATIENTS: SelectedPatient[] = [
  { id: 'p1', name: 'Alex Carter', displayId: 'P-2024-001' },
  { id: 'p2', name: 'Jordan Kim', displayId: 'P-2024-002' },
  { id: 'p3', name: 'Morgan Lee', displayId: 'P-2024-003' },
  { id: 'p4', name: 'Taylor Brooks', displayId: 'P-2024-004' },
  { id: 'p5', name: 'Casey Rivera', displayId: 'P-2024-005' },
  { id: 'p6', name: 'Jamie Patel', displayId: 'P-2024-006' },
  { id: 'p7', name: 'Avery Thompson', displayId: 'P-2024-007' },
  { id: 'p8', name: 'Riley Foster', displayId: 'P-2024-008' },
  { id: 'p9', name: 'Parker Nguyen', displayId: 'P-2024-009' },
  { id: 'p10', name: 'Skyler Adams', displayId: 'P-2024-010' },
  { id: 'p11', name: 'Drew Sullivan', displayId: 'P-2024-011' },
  { id: 'p12', name: 'Quinn Bailey', displayId: 'P-2024-012' }
];



const SelectPatients: React.FC = () => {
  const navigate = useNavigate();
  const [patients, setPatients] = useState<SelectedPatient[]>([]);
  const [loading, setLoading] = useState(true);
  const [usingDatabase, setUsingDatabase] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadPatients();
  }, []);

  const loadPatients = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const dbPatients = await PatientAPI.getAll();
      
      // Convert database patients to SelectedPatient format
      const formattedPatients: SelectedPatient[] = dbPatients.map((p: Patient) => ({
        id: p.id.toString(),
        name: p.name,
        displayId: `P-${p.id.toString().padStart(4, '0')}`,
      }));
      
      setPatients(formattedPatients);
      setUsingDatabase(true);
    } catch (err) {
      // Fall back to placeholder data if database is unavailable
      if (err instanceof APIError && err.statusCode === 503) {
        setError('Database not available. Using local mode.');
        setPatients(PLACEHOLDER_PATIENTS);
        setUsingDatabase(false);
      } else {
        setError('Failed to load patients. Using local mode.');
        setPatients(PLACEHOLDER_PATIENTS);
        setUsingDatabase(false);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleSelectPatient = (patient: SelectedPatient) => {
    if (usingDatabase) {
      // Navigate to patient management detail view for database patients
      navigate(`/patient-management/${patient.id}`);
    } else {
      // Use original behavior for placeholder patients
      navigate('/analysis', { state: { selectedPatient: patient } });
    }
  };

  if (loading) {
    return (
      <main className={styles.page}>
        <div className={styles.bgEffects} aria-hidden="true">
          <div className={styles.blurBlob1}></div>
          <div className={styles.blurBlob2}></div>
        </div>
        <section className={styles.card}>
          <div style={{ textAlign: 'center', padding: '2rem', color: '#94a3b8' }}>
            Loading patients...
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className={styles.page}>
      <div className={styles.bgEffects} aria-hidden="true">
        <div className={styles.blurBlob1}></div>
        <div className={styles.blurBlob2}></div>
      </div>

      <section className={styles.card} aria-label="Select Patients">
        <button
          type="button"
          className={styles.backButton}
          onClick={() => navigate('/menu')}
        >
          Back to Main Menu
        </button>

        <h1 className={styles.title}>
          {usingDatabase ? 'Select Patient (Database)' : 'Select Patient (Local Mode)'}
        </h1>

        {error && (
          <div style={{
            background: 'rgba(234, 179, 8, 0.1)',
            border: '1px solid rgba(234, 179, 8, 0.3)',
            color: '#fde047',
            padding: '0.75rem',
            borderRadius: '0.375rem',
            marginBottom: '1rem',
            fontSize: '0.9rem'
          }}>
            {error}
          </div>
        )}
        
        <div className={styles.list} role="list" aria-label="Patient list">
          {patients.map((patient) => (
            <div key={patient.id} className={styles.listRow} role="listitem">
              <div className={styles.patientMeta}>
                <span className={styles.patientName}>{patient.name}</span>
                <span className={styles.patientId}>{patient.displayId}</span>
              </div>
              <button
                type="button"
                className={styles.rowAction}
                onClick={() => handleSelectPatient(patient)}
                aria-label={`Select ${patient.name}`}
                title={`Select ${patient.name}`}
              >
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                  <path d="M6 3L11 8L6 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
};

export default SelectPatients;
