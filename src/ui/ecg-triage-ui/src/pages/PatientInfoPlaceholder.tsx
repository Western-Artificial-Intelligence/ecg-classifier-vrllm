import React from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/squarespace-theme.css';

const PatientInfoPlaceholder: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="squarespace-scope">
      <div className="ss-app-menu">
        <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
          <div className="ss-badge">
            Patient Information Module
          </div>
          <h1 className="ss-hero-title" style={{ fontSize: '3rem', marginBottom: '1.5rem' }}>
            Patient Info
          </h1>
          <p className="ss-hero-description" style={{ maxWidth: '700px', margin: '0 auto' }}>
            This module will allow you to view and manage patient demographics, 
            medical history, and contact information.
          </p>
        </div>

        <div className="glass-card" style={{ maxWidth: '800px', margin: '0 auto', textAlign: 'center', padding: '4rem 3rem' }}>
          <div style={{ fontSize: '4rem', fontWeight: '600', marginBottom: '2rem', color: '#000', opacity: '0.15' }}>UNDER DEVELOPMENT</div>
          <h2 style={{ fontSize: '2rem', fontWeight: '600', color: '#000', marginBottom: '1.5rem' }}>
            Coming Soon
          </h2>
          <p style={{ fontSize: '1rem', color: 'rgba(0, 0, 0, 0.7)', lineHeight: '1.7', marginBottom: '2rem' }}>
            The Patient Information module is currently under development. This feature will include:
          </p>
          
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(2, 1fr)', 
            gap: '1.5rem',
            marginBottom: '3rem',
            textAlign: 'left'
          }}>
            <div style={{ 
              padding: '1.5rem', 
              background: 'rgba(0, 0, 0, 0.03)', 
              borderRadius: '8px',
              border: '1px solid rgba(0, 0, 0, 0.08)'
            }}>
              <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem', fontWeight: '600', color: 'rgba(0, 0, 0, 0.3)' }}>Demographics</div>
              <h3 style={{ fontSize: '1rem', fontWeight: '600', color: '#000', marginBottom: '0.5rem' }}>
                Patient Demographics
              </h3>
              <p style={{ fontSize: '0.875rem', color: 'rgba(0, 0, 0, 0.6)' }}>
                Age, gender, height, weight, BMI
              </p>
            </div>

            <div style={{ 
              padding: '1.5rem', 
              background: 'rgba(0, 0, 0, 0.03)', 
              borderRadius: '8px',
              border: '1px solid rgba(0, 0, 0, 0.08)'
            }}>
              <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem', fontWeight: '600', color: 'rgba(0, 0, 0, 0.3)' }}>History</div>
              <h3 style={{ fontSize: '1rem', fontWeight: '600', color: '#000', marginBottom: '0.5rem' }}>
                Medical History
              </h3>
              <p style={{ fontSize: '0.875rem', color: 'rgba(0, 0, 0, 0.6)' }}>
                Conditions, medications, allergies
              </p>
            </div>

            <div style={{ 
              padding: '1.5rem', 
              background: 'rgba(0, 0, 0, 0.03)', 
              borderRadius: '8px',
              border: '1px solid rgba(0, 0, 0, 0.08)'
            }}>
              <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem', fontWeight: '600', color: 'rgba(0, 0, 0, 0.3)' }}>Contact</div>
              <h3 style={{ fontSize: '1rem', fontWeight: '600', color: '#000', marginBottom: '0.5rem' }}>
                Contact Information
              </h3>
              <p style={{ fontSize: '0.875rem', color: 'rgba(0, 0, 0, 0.6)' }}>
                Phone, email, emergency contacts
              </p>
            </div>

            <div style={{ 
              padding: '1.5rem', 
              background: 'rgba(0, 0, 0, 0.03)', 
              borderRadius: '8px',
              border: '1px solid rgba(0, 0, 0, 0.08)'
            }}>
              <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem', fontWeight: '600', color: 'rgba(0, 0, 0, 0.3)' }}>Risk</div>
              <h3 style={{ fontSize: '1rem', fontWeight: '600', color: '#000', marginBottom: '0.5rem' }}>
                Risk Factors
              </h3>
              <p style={{ fontSize: '0.875rem', color: 'rgba(0, 0, 0, 0.6)' }}>
                Sleep apnea risk assessment scores
              </p>
            </div>
          </div>

          <p style={{ 
            padding: '1rem 1.5rem', 
            background: 'rgba(0, 0, 0, 0.03)',
            border: '1px solid rgba(0, 0, 0, 0.08)',
            borderRadius: '6px',
            color: 'rgba(0, 0, 0, 0.7)',
            marginBottom: '2rem',
            fontSize: '0.875rem',
            lineHeight: '1.6'
          }}>
            <strong>Tip:</strong> For now, you can explore the Patient Analysis module 
            to see ECG analysis and sleep apnea detection features.
          </p>

          <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
            <button 
              className="ss-back-button"
              onClick={() => navigate('/app')}
            >
              Back to Menu
            </button>
            <button 
              className="cta-orange"
              onClick={() => navigate('/app/analysis')}
            >
              Try Patient Analysis
            </button>
          </div>
        </div>

        <div style={{ textAlign: 'center', marginTop: '3rem' }}>
          <button 
            className="ss-back-button"
            onClick={() => navigate('/')}
          >
            Back to Home
          </button>
        </div>
      </div>
    </div>
  );
};

export default PatientInfoPlaceholder;
