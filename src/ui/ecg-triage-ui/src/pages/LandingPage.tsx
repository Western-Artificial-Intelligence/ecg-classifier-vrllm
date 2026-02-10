import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/squarespace-theme.css';

const LandingPage: React.FC = () => {
  const navigate = useNavigate();
  const [showDemoModal, setShowDemoModal] = useState(false);

  return (
    <div className="squarespace-scope">
      {/* Navigation */}
      <nav className="ss-nav">
        <div className="ss-nav-content">
            <a href="/" className="ss-logo">
            <span>NeuralApnea Triage</span>
          </a>
          <div className="ss-nav-links">
            <a href="#about" className="ss-nav-link">About</a>
            <a href="#features" className="ss-nav-link">Features</a>
            <a href="#how-it-works" className="ss-nav-link">How It Works</a>
            <a href="#paper" className="ss-nav-link">Paper</a>
            <button onClick={() => navigate('/app')} className="cta-orange">
              Get started
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="ss-hero">
        <div className="ss-hero-content">
          <div className="ss-badge">
            AI-Powered Sleep Apnea Screening
          </div>
          
          <h1 className="ss-hero-title">
            Transform Sleep Apnea Detection with <span className="gradient-text">AI & ECG</span>
          </h1>
          
          <p className="ss-hero-description">
            NeuralApnea Triage uses advanced machine learning to analyze overnight ECG recordings, 
            providing fast, accessible screening for sleep apnea—helping prioritize patients who 
            need polysomnography and reducing diagnostic delays.
          </p>
          
          <div className="ss-hero-actions">
            <button onClick={() => navigate('/app')} className="cta-orange">
              Get started
            </button>
            <button onClick={() => setShowDemoModal(true)} className="btn-secondary">
              Watch demo
            </button>
          </div>
        </div>

        <div className="ss-hero-visual">
          <div className="glass-card">
            <div style={{ marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h3 style={{ fontSize: '1.125rem', fontWeight: '600', color: '#000', margin: 0 }}>ECG Analysis</h3>
              <div style={{ 
                padding: '0.375rem 0.75rem', 
                background: 'rgba(0, 0, 0, 0.05)', 
                borderRadius: '4px',
                border: '1px solid rgba(0, 0, 0, 0.08)',
                color: 'rgba(0, 0, 0, 0.7)', 
                fontSize: '0.8125rem', 
                fontWeight: '500'
              }}>
                Processing
              </div>
            </div>
            
            <div style={{ 
              padding: '1.5rem', 
              background: 'rgba(0, 0, 0, 0.02)', 
              borderRadius: '8px', 
              marginBottom: '1.5rem',
              border: '1px solid rgba(0, 0, 0, 0.06)',
              height: '120px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'rgba(0, 0, 0, 0.3)',
              fontSize: '0.875rem',
              fontWeight: '500'
            }}>
              ECG Waveform Visualization
            </div>
            
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(3, 1fr)', 
              gap: '1rem' 
            }}>
              <div style={{ 
                padding: '1rem', 
                background: 'rgba(0, 0, 0, 0.03)', 
                borderRadius: '6px',
                border: '1px solid rgba(0, 0, 0, 0.06)',
                textAlign: 'center'
              }}>
                <div style={{ fontSize: '0.6875rem', color: 'rgba(0, 0, 0, 0.6)', marginBottom: '0.25rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Risk Level</div>
                <div style={{ fontSize: '1.25rem', fontWeight: '600', color: '#000' }}>High</div>
              </div>
              <div style={{ 
                padding: '1rem', 
                background: 'rgba(0, 0, 0, 0.03)', 
                borderRadius: '6px',
                border: '1px solid rgba(0, 0, 0, 0.06)',
                textAlign: 'center'
              }}>
                <div style={{ fontSize: '0.6875rem', color: 'rgba(0, 0, 0, 0.6)', marginBottom: '0.25rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Confidence</div>
                <div style={{ fontSize: '1.25rem', fontWeight: '600', color: '#000' }}>94%</div>
              </div>
              <div style={{ 
                padding: '1rem', 
                background: 'rgba(0, 0, 0, 0.03)', 
                borderRadius: '6px',
                border: '1px solid rgba(0, 0, 0, 0.06)',
                textAlign: 'center'
              }}>
                <div style={{ fontSize: '0.6875rem', color: 'rgba(0, 0, 0, 0.6)', marginBottom: '0.25rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Events</div>
                <div style={{ fontSize: '1.25rem', fontWeight: '600', color: '#000' }}>12</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="ss-section">
        <div className="ss-stats-grid">
          <div className="ss-stat-card glass-card">
            <div className="ss-stat-number">80%</div>
            <div className="ss-stat-label">Sleep Apnea Cases Undiagnosed</div>
          </div>
          <div className="ss-stat-card glass-card">
            <div className="ss-stat-number">8-30mo</div>
            <div className="ss-stat-label">Wait Time for PSG in Canada</div>
          </div>
          <div className="ss-stat-card glass-card">
            <div className="ss-stat-number">$1-10K</div>
            <div className="ss-stat-label">Cost Per Polysomnography Study</div>
          </div>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className="ss-section">
        <div className="ss-section-badge">About the Platform</div>
        <h2 className="ss-section-title">AI-Powered Triage for Better Access to Care</h2>
        <p className="ss-section-description">
          Sleep apnea affects hundreds of millions globally, yet approximately 80% of cases remain 
          undiagnosed. Traditional diagnosis through polysomnography (PSG) is expensive, time-intensive, 
          and suffers from severe accessibility issues. NeuralApnea Triage leverages ECG-based machine 
          learning to screen patients efficiently, helping healthcare providers prioritize limited 
          diagnostic resources and reduce care delays.
        </p>

        <div className="ss-feature-grid">
          <div className="glass-card">
            <div className="ss-feature-icon">Security</div>
            <h3 className="ss-feature-title">Clinical Screening Tool</h3>
            <p className="ss-feature-description">
              A demonstration platform showcasing AI-powered triage for sleep apnea. Analyzes ECG patterns 
              to identify high-risk patients who should be prioritized for polysomnography evaluation.
            </p>
          </div>

          <div className="glass-card">
            <div className="ss-feature-icon">Insight</div>
            <h3 className="ss-feature-title">Explainable AI</h3>
            <p className="ss-feature-description">
              Every prediction includes interpretable biomarkers—HRV metrics, sample entropy, attention 
              visualizations—so clinicians understand the physiological signals driving each risk assessment.
            </p>
          </div>

          <div className="glass-card">
            <div className="ss-feature-icon">Speed</div>
            <h3 className="ss-feature-title">ECG-Based Screening</h3>
            <p className="ss-feature-description">
              Works with overnight single-lead ECG data already collected during routine monitoring. 
              No additional sensors or specialized workflows required—unlocking insights from existing data.
            </p>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="ss-section">
        <div className="ss-section-badge">Core Capabilities</div>
        <h2 className="ss-section-title">Powerful Features for Clinical Insight</h2>
        
        <div className="ss-feature-grid">
          <div className="glass-card">
            <div className="ss-feature-icon">AI</div>
            <h3 className="ss-feature-title">Hybrid CNN-Transformer Architecture</h3>
            <p className="ss-feature-description">
              Combines convolutional layers for local pattern detection with transformer attention 
              mechanisms to capture long-range temporal dependencies in heart rate variability.
            </p>
          </div>

          <div className="glass-card">
            <div className="ss-feature-icon">Analysis</div>
            <h3 className="ss-feature-title">Risk Stratification & Triage</h3>
            <p className="ss-feature-description">
              Automatically categorizes patients into Low, Medium, or High risk tiers with confidence scores, 
              helping allocate limited PSG resources to those most likely to benefit.
            </p>
          </div>

          <div className="glass-card">
            <div className="ss-feature-icon">Time</div>
            <h3 className="ss-feature-title">Segment-Level Analysis</h3>
            <p className="ss-feature-description">
              Pinpoints specific time windows where apnea events are suspected, enabling targeted 
              clinical review and aggregated apnea burden scoring across overnight recordings.
            </p>
          </div>

          <div className="glass-card">
            <div className="ss-feature-icon">Data</div>
            <h3 className="ss-feature-title">Physiological Biomarkers</h3>
            <p className="ss-feature-description">
              Explainability layer surfaces interpretable metrics like VLF/HF ratio, sample entropy, 
              and R-peak amplitude variability—correlating with intermittent hypoxia patterns.
            </p>
          </div>

          <div className="glass-card">
            <div className="ss-feature-icon">Visual</div>
            <h3 className="ss-feature-title">Grad-CAM Visualization</h3>
            <p className="ss-feature-description">
              Highlights the exact ECG regions that influenced model predictions, providing visual 
              explanations that build clinician trust and enable validation.
            </p>
          </div>

          <div className="glass-card">
            <div className="ss-feature-icon">Scale</div>
            <h3 className="ss-feature-title">Scalable & Accessible</h3>
            <p className="ss-feature-description">
              Software-only solution that works with existing ECG infrastructure, enabling deployment 
              in resource-constrained settings without additional hardware costs.
            </p>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="ss-section">
        <div className="ss-section-badge">How It Works</div>
        <h2 className="ss-section-title">Simple Workflow, Powerful Results</h2>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '2rem', marginTop: '4rem' }}>
          <div className="glass-card" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '2.5rem', fontWeight: '600', marginBottom: '1rem', color: '#000', opacity: '0.15' }}>01</div>
            <h3 className="ss-feature-title" style={{ fontSize: '1.125rem' }}>Upload ECG</h3>
            <p className="ss-feature-description">
              Upload overnight single-lead ECG recording in standard format
            </p>
          </div>

          <div className="glass-card" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '2.5rem', fontWeight: '600', marginBottom: '1rem', color: '#000', opacity: '0.15' }}>02</div>
            <h3 className="ss-feature-title" style={{ fontSize: '1.125rem' }}>AI Analysis</h3>
            <p className="ss-feature-description">
              Model analyzes HRV patterns to detect potential apnea events
            </p>
          </div>

          <div className="glass-card" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '2.5rem', fontWeight: '600', marginBottom: '1rem', color: '#000', opacity: '0.15' }}>03</div>
            <h3 className="ss-feature-title" style={{ fontSize: '1.125rem' }}>Review Results</h3>
            <p className="ss-feature-description">
              View risk level, flagged segments, and explanations
            </p>
          </div>

          <div className="glass-card" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '2.5rem', fontWeight: '600', marginBottom: '1rem', color: '#000', opacity: '0.15' }}>04</div>
            <h3 className="ss-feature-title" style={{ fontSize: '1.125rem' }}>Take Action</h3>
            <p className="ss-feature-description">
              Use insights to prioritize PSG referrals and follow-up
            </p>
          </div>
        </div>
      </section>

      {/* Paper Section */}
      <section id="paper" className="ss-section">
        <div className="ss-section-badge">Research</div>
        <h2 className="ss-section-title">Academic Research</h2>
        <p className="ss-section-description">
          Our work explores CNN-Transformer architectures for ECG-based sleep apnea detection, 
          combining convolutional feature extraction with attention mechanisms to capture 
          long-range temporal dependencies in heart rate variability patterns.
        </p>
        
        <div className="glass-card" style={{ textAlign: 'center', padding: '4rem' }}>
          <div style={{ fontSize: '4rem', fontWeight: '600', marginBottom: '2rem', color: '#000', opacity: '0.15' }}>PAPER</div>
          <h3 className="ss-feature-title" style={{ fontSize: '1.75rem', marginBottom: '1rem' }}>
            Paper Coming Soon
          </h3>
          <p className="ss-feature-description" style={{ fontSize: '1rem' }}>
            PDF will be embedded here once our research paper is published.
          </p>
          <p style={{ color: 'rgba(0, 0, 0, 0.6)', marginTop: '2rem', fontStyle: 'italic', fontSize: '0.875rem' }}>
            Working Title: AgenticCardioGram: Machine Learning Powered ECG Analysis System for Sleep Apnea Classification
          </p>
        </div>
      </section>

      {/* Team Section */}
      <section className="ss-section">
        <div className="ss-section-badge">Our Team</div>
        <h2 className="ss-section-title">Built by Researchers at Western University</h2>
        <p className="ss-section-description">
          Developed by a multidisciplinary team for the CUCAI competition, combining expertise 
          in machine learning, signal processing, and clinical medicine.
        </p>
        
        <div className="ss-team-grid">
          <div className="ss-team-member glass-card">
            <div style={{ width: '64px', height: '64px', borderRadius: '50%', background: 'rgba(0, 0, 0, 0.05)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.25rem', fontWeight: '600', margin: '0 auto 1rem', border: '1px solid rgba(0, 0, 0, 0.08)', color: 'rgba(0, 0, 0, 0.5)' }}>OO</div>
            <h3 className="ss-team-name">Oliver Olejar</h3>
            <p className="ss-team-role">Western University</p>
          </div>
          <div className="ss-team-member glass-card">
            <div style={{ width: '64px', height: '64px', borderRadius: '50%', background: 'rgba(0, 0, 0, 0.05)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.25rem', fontWeight: '600', margin: '0 auto 1rem', border: '1px solid rgba(0, 0, 0, 0.08)', color: 'rgba(0, 0, 0, 0.5)' }}>DK</div>
            <h3 className="ss-team-name">Daniel Kaminsky</h3>
            <p className="ss-team-role">Western University</p>
          </div>
          <div className="ss-team-member glass-card">
            <div style={{ width: '64px', height: '64px', borderRadius: '50%', background: 'rgba(0, 0, 0, 0.05)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.25rem', fontWeight: '600', margin: '0 auto 1rem', border: '1px solid rgba(0, 0, 0, 0.08)', color: 'rgba(0, 0, 0, 0.5)' }}>AL</div>
            <h3 className="ss-team-name">Annie Liu</h3>
            <p className="ss-team-role">Western University</p>
          </div>
          <div className="ss-team-member glass-card">
            <div style={{ width: '64px', height: '64px', borderRadius: '50%', background: 'rgba(0, 0, 0, 0.05)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.25rem', fontWeight: '600', margin: '0 auto 1rem', border: '1px solid rgba(0, 0, 0, 0.08)', color: 'rgba(0, 0, 0, 0.5)' }}>JM</div>
            <h3 className="ss-team-name">John MacPhie</h3>
            <p className="ss-team-role">Western University</p>
          </div>
          <div className="ss-team-member glass-card">
            <div style={{ width: '64px', height: '64px', borderRadius: '50%', background: 'rgba(0, 0, 0, 0.05)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.25rem', fontWeight: '600', margin: '0 auto 1rem', border: '1px solid rgba(0, 0, 0, 0.08)', color: 'rgba(0, 0, 0, 0.5)' }}>SS</div>
            <h3 className="ss-team-name">Sneha Shah</h3>
            <p className="ss-team-role">Western University</p>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="ss-section">
        <div className="glass-card" style={{ 
          display: 'flex', 
          flexDirection: 'column', 
          alignItems: 'center', 
          justifyContent: 'center',
          textAlign: 'center', 
          padding: '5rem 3rem'
        }}>
          <h2 className="ss-section-title" style={{ 
            marginBottom: '1rem'
          }}>
            Ready to Transform Sleep Apnea Screening?
          </h2>
          <p className="ss-section-description" style={{ 
            marginBottom: '2.5rem', 
            fontSize: '1.125rem'
          }}>
            Start analyzing ECG data with AI-powered insights today.
          </p>
          <button onClick={() => navigate('/app')} className="cta-orange" style={{ fontSize: '1.0625rem', padding: '1rem 2rem' }}>
            Get started
          </button>
        </div>
      </section>

      {/* Footer */}
      <footer className="ss-footer">
        <div className="ss-footer-content">
          <div className="ss-footer-disclaimer">
            <strong>Medical Disclaimer:</strong> NeuralApnea Triage is a research prototype and demonstration 
            tool built for educational purposes and the CUCAI competition. It is NOT a medical device and is 
            NOT intended for clinical diagnosis or treatment decisions. This system is not a replacement 
            for polysomnography (PSG). Always consult with qualified healthcare professionals for medical 
            decisions and diagnosis. This tool demonstrates the potential of AI-powered ECG screening for 
            sleep apnea triage support.
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '2rem' }}>
            <div className="ss-logo">
              <span>NeuralApnea Triage</span>
            </div>

            <div className="ss-footer-links">
              <a href="#about" className="ss-footer-link">About</a>
              <a href="#features" className="ss-footer-link">Features</a>
              <a href="#how-it-works" className="ss-footer-link">How It Works</a>
              <a href="#paper" className="ss-footer-link">Research</a>
              <a href="#" className="ss-footer-link">Privacy</a>
              <a href="#" className="ss-footer-link">Contact</a>
            </div>
          </div>

          <div className="ss-copyright">
            © 2026 NeuralApnea Triage. Built for CUCAI Competition by Oliver Olejar, Daniel Kaminsky, Annie Liu, John MacPhie, and Sneha Shah.
          </div>
        </div>
      </footer>

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
              Demo video coming soon. For now, explore the platform by clicking "Get started" to see the 
              patient analysis interface.
            </p>
            <button onClick={() => setShowDemoModal(false)} className="cta-orange">
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default LandingPage;
