import React from 'react';
import { useNavigate } from 'react-router-dom';
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

  const handleSelectPatient = (patient: SelectedPatient) => {
    navigate('/analysis', { state: { selectedPatient: patient } });
  };

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

        <h1 className={styles.title}>Select Patient</h1>

        <div className={styles.list} role="list" aria-label="Patient list">
          {PLACEHOLDER_PATIENTS.map((patient) => (
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
