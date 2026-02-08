import React from 'react';
import styles from '../../styles/Explainability.module.css';

const Explainability: React.FC = () => {
  return (
    <section id="explainability" className={styles.explainability}>
      <div className={styles.explainabilityContainer}>
        <div className={styles.sectionLabel}>Explainability Agent</div>
        <h2 className={styles.title}>Understanding the "why" behind predictions</h2>
        <p className={styles.subtitle}>
          Clinical trust requires transparency. Our explainability agent generates 
          human-readable summaries and visual highlights of apnea-related patterns.
        </p>

        <div className={styles.techniques}>
          <div className={styles.techniqueCard}>
            <div className={styles.techniqueIcon}>
              <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                <path d="M6 16h4l3-8 4 16 4-12 3 4h6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </div>
            <h3 className={styles.techniqueTitle}>HRV Feature Summaries</h3>
            <p className={styles.techniqueDescription}>
              Extracts interpretable biomarkers like VLF/HF ratio, sample entropy, and 
              RMSSD to explain model decisions in clinically meaningful terms.
            </p>
            <div className={styles.techniqueExample}>
              <div className={styles.exampleLabel}>Example Output:</div>
              <div className={styles.exampleText}>
                "Elevated VLF/HF ratio (2.3 → 4.1) during 02:30–02:35 suggests sympathetic 
                activation consistent with apnea arousal."
              </div>
            </div>
          </div>

          <div className={styles.techniqueCard}>
            <div className={styles.techniqueIcon}>
              <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                <rect x="4" y="4" width="24" height="24" rx="2" stroke="currentColor" strokeWidth="2" fill="none"/>
                <path d="M4 12h24M12 4v24" stroke="currentColor" strokeWidth="2"/>
              </svg>
            </div>
            <h3 className={styles.techniqueTitle}>Grad-CAM-style Highlighting</h3>
            <p className={styles.techniqueDescription}>
              Visualizes attention-weighted regions in the ECG timeline where the model 
              focused most heavily—highlighting likely apnea events for clinician review.
            </p>
            <div className={styles.techniqueExample}>
              <div className={styles.exampleLabel}>Visual Output:</div>
              <div className={styles.heatmapMock}>
                <div className={styles.heatmapBar}>
                  <div className={styles.heatmapSegment} style={{width: '20%', opacity: 0.3}}></div>
                  <div className={styles.heatmapSegment} style={{width: '15%', opacity: 0.9}}></div>
                  <div className={styles.heatmapSegment} style={{width: '25%', opacity: 0.4}}></div>
                  <div className={styles.heatmapSegment} style={{width: '18%', opacity: 0.8}}></div>
                  <div className={styles.heatmapSegment} style={{width: '22%', opacity: 0.2}}></div>
                </div>
                <div className={styles.heatmapLabels}>
                  <span>00:00</span>
                  <span>02:00</span>
                  <span>04:00</span>
                  <span>06:00</span>
                  <span>08:00</span>
                </div>
                <div className={styles.heatmapLegend}>
                  <span>Low attention</span>
                  <div className={styles.gradientBar}></div>
                  <span>High attention</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className={styles.benefits}>
          <h3 className={styles.benefitsTitle}>Why explainability matters</h3>
          <div className={styles.benefitsGrid}>
            <div className={styles.benefitItem}>
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                <circle cx="10" cy="10" r="7" stroke="currentColor" strokeWidth="2" fill="none"/>
                <path d="M7 10l2 2 4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              <span>Builds clinician trust and confidence</span>
            </div>
            <div className={styles.benefitItem}>
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                <circle cx="10" cy="10" r="7" stroke="currentColor" strokeWidth="2" fill="none"/>
                <path d="M7 10l2 2 4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              <span>Enables clinical validation of predictions</span>
            </div>
            <div className={styles.benefitItem}>
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                <circle cx="10" cy="10" r="7" stroke="currentColor" strokeWidth="2" fill="none"/>
                <path d="M7 10l2 2 4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              <span>Facilitates model debugging and improvement</span>
            </div>
            <div className={styles.benefitItem}>
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                <circle cx="10" cy="10" r="7" stroke="currentColor" strokeWidth="2" fill="none"/>
                <path d="M7 10l2 2 4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              <span>Meets regulatory requirements for clinical AI</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Explainability;

