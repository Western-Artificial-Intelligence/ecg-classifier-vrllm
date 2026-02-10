import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/squarespace-theme.css';

const AppMainMenu: React.FC = () => {
  const navigate = useNavigate();
  const [showDemoModal, setShowDemoModal] = useState(false);

  return (
    <div className="squarespace-scope">
      <div className="ss-app-menu">
        <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
          <div className="ss-badge">
            NeuralApnea Triage Platform
          </div>
          <h1 className="ss-hero-title" style={{ fontSize: '3rem', marginBottom: '1.5rem' }}>
            Select a Module
          </h1>
          <p className="ss-hero-description" style={{ maxWidth: '700px', margin: '0 auto' }}>
            Choose from the available modules below to access patient information, 
            run ECG analysis, or watch a demonstration of the platform.
          </p>
        </div>

        <div className="ss-menu-cards">
          <div 
            className="ss-menu-card"
            onClick={() => navigate('/app/patient')}
          >
            <div className="ss-menu-icon">Patient Info</div>
            <h3 className="ss-menu-title">Patient Info</h3>
            <p className="ss-menu-description">
              View and manage patient demographics, medical history, and contact information.
            </p>
            <div style={{ 
              marginTop: 'auto', 
              padding: '0.625rem 1.25rem', 
              background: 'rgba(0, 0, 0, 0.05)', 
              borderRadius: '4px',
              fontSize: '0.8125rem',
              fontWeight: '500',
              color: 'rgba(0, 0, 0, 0.7)'
            }}>
              Coming Soon
            </div>
          </div>

          <div 
            className="ss-menu-card"
            onClick={() => navigate('/app/analysis')}
          >
            <div className="ss-menu-icon">Analysis</div>
            <h3 className="ss-menu-title">Patient Analysis</h3>
            <p className="ss-menu-description">
              Analyze overnight ECG recordings with AI-powered sleep apnea detection and explainability.
            </p>
            <button 
              className="cta-orange"
              style={{ marginTop: 'auto' }}
              onClick={(e) => {
                e.stopPropagation();
                navigate('/app/analysis');
              }}
            >
              Launch Analysis
            </button>
          </div>

          <div 
            className="ss-menu-card"
            onClick={() => setShowDemoModal(true)}
          >
            <div className="ss-menu-icon">Demo</div>
            <h3 className="ss-menu-title">Watch Demo</h3>
            <p className="ss-menu-description">
              See a walkthrough of the platform's features and capabilities in action.
            </p>
            <button 
              className="btn-secondary"
              style={{ marginTop: 'auto' }}
              onClick={(e) => {
                e.stopPropagation();
                setShowDemoModal(true);
              }}
            >
              Play Demo
            </button>
          </div>
        </div>

        <div style={{ textAlign: 'center', marginTop: '4rem' }}>
          <button 
            className="ss-back-button"
            onClick={() => navigate('/')}
          >
            Back to Home
          </button>
        </div>
      </div>

      {/* Demo Modal */}
      {showDemoModal && (
        <div 
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.6)',
            backdropFilter: 'blur(8px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
            padding: '2rem'
          }}
          onClick={() => setShowDemoModal(false)}
        >
          <div 
            className="glass-card"
            style={{ maxWidth: '500px', textAlign: 'center' }}
            onClick={(e) => e.stopPropagation()}
          >
            <h3 style={{ fontSize: '1.75rem', color: '#000', marginBottom: '1rem', fontWeight: '600' }}>Demo Video</h3>
            <p style={{ color: 'rgba(0, 0, 0, 0.7)', marginBottom: '2rem', lineHeight: '1.7' }}>
              Demo video coming soon. For now, explore the Patient Analysis module to see 
              ECG analysis, risk stratification, and explainability features in action.
            </p>
            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
              <button onClick={() => setShowDemoModal(false)} className="btn-secondary">
                Close
              </button>
              <button 
                onClick={() => {
                  setShowDemoModal(false);
                  navigate('/app/analysis');
                }} 
                className="cta-orange"
              >
                Try Patient Analysis
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AppMainMenu;
