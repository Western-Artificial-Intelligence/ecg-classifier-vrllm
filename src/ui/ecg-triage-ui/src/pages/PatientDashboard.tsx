import React from 'react';
import { useNavigate } from 'react-router-dom';
import styles from '../styles/PatientDashboard.module.css';

interface Patient {
  id: string;
  name: string;
  recordId: string;
  age: number;
  sex: string;
  studyDate: string;
  riskLevel: 'Low' | 'Medium' | 'High';
  confidence: number;
  status: 'Analyzed' | 'Pending' | 'In Progress';
}

const mockPatients: Patient[] = [
  {
    id: 'a01',
    name: 'Patient A01',
    recordId: 'a01',
    age: 52,
    sex: 'M',
    studyDate: '2026-02-05',
    riskLevel: 'High',
    confidence: 94,
    status: 'Analyzed'
  },
  {
    id: 'a02',
    name: 'Patient A02',
    recordId: 'a02',
    age: 45,
    sex: 'F',
    studyDate: '2026-02-04',
    riskLevel: 'Low',
    confidence: 89,
    status: 'Analyzed'
  },
  {
    id: 'a03',
    name: 'Patient A03',
    recordId: 'a03',
    age: 61,
    sex: 'M',
    studyDate: '2026-02-03',
    riskLevel: 'High',
    confidence: 91,
    status: 'Analyzed'
  },
  {
    id: 'a04',
    name: 'Patient A04',
    recordId: 'a04',
    age: 38,
    sex: 'F',
    studyDate: '2026-02-02',
    riskLevel: 'Medium',
    confidence: 76,
    status: 'Analyzed'
  },
  {
    id: 'b01',
    name: 'Patient B01',
    recordId: 'b01',
    age: 58,
    sex: 'M',
    studyDate: '2026-02-01',
    riskLevel: 'Low',
    confidence: 92,
    status: 'Analyzed'
  },
  {
    id: 'c01',
    name: 'Patient C01',
    recordId: 'c01',
    age: 47,
    sex: 'F',
    studyDate: '2026-01-31',
    riskLevel: 'Medium',
    confidence: 82,
    status: 'Analyzed'
  }
];

const PatientDashboard: React.FC = () => {
  const navigate = useNavigate();

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'High':
        return styles.riskHigh;
      case 'Medium':
        return styles.riskMedium;
      case 'Low':
        return styles.riskLow;
      default:
        return '';
    }
  };

  const handlePatientClick = (patientId: string) => {
    navigate(`/analysis/${patientId}`);
  };

  const handleBackToHome = () => {
    navigate('/');
  };

  return (
    <div className={styles.dashboardContainer}>
      {/* Header */}
      <header className={styles.header}>
        <div className={styles.headerContent}>
          <div className={styles.logo}>
            <div className={styles.logoIcon}>
              <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M4 16L8 16L11 8L15 24L19 12L22 16L28 16" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                <circle cx="16" cy="16" r="14" stroke="currentColor" strokeWidth="2" fill="none"/>
              </svg>
            </div>
            <span className={styles.logoText}>NeuralApnea Triage</span>
          </div>
          <button onClick={handleBackToHome} className={styles.backButton}>
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path d="M15 10H5M9 6l-4 4 4 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
            Back to Home
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className={styles.mainContent}>
        <div className={styles.contentWrapper}>
          {/* Page Title */}
          <div className={styles.pageHeader}>
            <h1 className={styles.pageTitle}>Patient Dashboard</h1>
            <p className={styles.pageDescription}>
              Select a patient record to view detailed ECG analysis and apnea risk assessment
            </p>
          </div>

          {/* Stats Overview */}
          <div className={styles.statsOverview}>
            <div className={styles.statCard}>
              <div className={styles.statIcon}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                  <path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2M9 11a4 4 0 100-8 4 4 0 000 8zM23 21v-2a4 4 0 00-3-3.87M16 3.13a4 4 0 010 7.75" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                </svg>
              </div>
              <div className={styles.statInfo}>
                <div className={styles.statValue}>{mockPatients.length}</div>
                <div className={styles.statLabel}>Total Patients</div>
              </div>
            </div>
            <div className={styles.statCard}>
              <div className={`${styles.statIcon} ${styles.highRiskIcon}`}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                  <path d="M12 9v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                </svg>
              </div>
              <div className={styles.statInfo}>
                <div className={styles.statValue}>
                  {mockPatients.filter(p => p.riskLevel === 'High').length}
                </div>
                <div className={styles.statLabel}>High Risk</div>
              </div>
            </div>
            <div className={styles.statCard}>
              <div className={`${styles.statIcon} ${styles.mediumRiskIcon}`}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                  <path d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                </svg>
              </div>
              <div className={styles.statInfo}>
                <div className={styles.statValue}>
                  {mockPatients.filter(p => p.riskLevel === 'Medium').length}
                </div>
                <div className={styles.statLabel}>Medium Risk</div>
              </div>
            </div>
            <div className={styles.statCard}>
              <div className={`${styles.statIcon} ${styles.lowRiskIcon}`}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                  <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                </svg>
              </div>
              <div className={styles.statInfo}>
                <div className={styles.statValue}>
                  {mockPatients.filter(p => p.riskLevel === 'Low').length}
                </div>
                <div className={styles.statLabel}>Low Risk</div>
              </div>
            </div>
          </div>

          {/* Patient List */}
          <div className={styles.patientListSection}>
            <div className={styles.sectionHeader}>
              <h2 className={styles.sectionTitle}>Patient Records</h2>
              <button className={styles.uploadButton}>
                <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                  <path d="M10 4v12M4 10h12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                </svg>
                Upload New ECG
              </button>
            </div>

            <div className={styles.patientTable}>
              <div className={styles.tableHeader}>
                <div className={styles.headerCell}>Patient</div>
                <div className={styles.headerCell}>Record ID</div>
                <div className={styles.headerCell}>Age/Sex</div>
                <div className={styles.headerCell}>Study Date</div>
                <div className={styles.headerCell}>Risk Level</div>
                <div className={styles.headerCell}>Confidence</div>
                <div className={styles.headerCell}>Status</div>
                <div className={styles.headerCell}>Action</div>
              </div>
              <div className={styles.tableBody}>
                {mockPatients.map((patient) => (
                  <div key={patient.id} className={styles.tableRow}>
                    <div className={styles.tableCell}>
                      <div className={styles.patientInfo}>
                        <div className={styles.patientAvatar}>
                          {patient.name.charAt(patient.name.length - 3)}
                        </div>
                        <span className={styles.patientName}>{patient.name}</span>
                      </div>
                    </div>
                    <div className={styles.tableCell}>
                      <span className={styles.recordId}>{patient.recordId}</span>
                    </div>
                    <div className={styles.tableCell}>
                      {patient.age} / {patient.sex}
                    </div>
                    <div className={styles.tableCell}>{patient.studyDate}</div>
                    <div className={styles.tableCell}>
                      <span className={`${styles.riskBadge} ${getRiskColor(patient.riskLevel)}`}>
                        {patient.riskLevel}
                      </span>
                    </div>
                    <div className={styles.tableCell}>
                      <div className={styles.confidenceBar}>
                        <div 
                          className={styles.confidenceFill} 
                          style={{ width: `${patient.confidence}%` }}
                        ></div>
                        <span className={styles.confidenceText}>{patient.confidence}%</span>
                      </div>
                    </div>
                    <div className={styles.tableCell}>
                      <span className={styles.statusBadge}>{patient.status}</span>
                    </div>
                    <div className={styles.tableCell}>
                      <button 
                        onClick={() => handlePatientClick(patient.recordId)}
                        className={styles.viewButton}
                      >
                        View Analysis
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default PatientDashboard;

