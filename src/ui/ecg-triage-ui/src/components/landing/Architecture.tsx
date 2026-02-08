import React from 'react';
import styles from '../../styles/Architecture.module.css';

const Architecture: React.FC = () => {
  return (
    <section id="architecture" className={styles.architecture}>
      <div className={styles.architectureContainer}>
        <div className={styles.sectionLabel}>System Architecture</div>
        <h2 className={styles.title}>From raw ECG to interpretable predictions</h2>
        
        <div className={styles.pipeline}>
          <div className={styles.step}>
            <div className={styles.stepNumber}>01</div>
            <div className={styles.stepContent}>
              <h3 className={styles.stepTitle}>Preprocess ECG</h3>
              <p className={styles.stepDescription}>
                Extract R-peaks from raw single-lead ECG signal using Pan-Tompkins 
                algorithm and bandpass filtering.
              </p>
              <div className={styles.stepDetails}>
                <div className={styles.detail}>
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                    <path d="M8 3v10M3 8h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                  </svg>
                  <span>Bandpass filter (5–15 Hz)</span>
                </div>
                <div className={styles.detail}>
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                    <path d="M8 3v10M3 8h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                  </svg>
                  <span>R-peak detection</span>
                </div>
              </div>
            </div>
          </div>

          <div className={styles.arrow}>
            <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
              <path d="M10 20h20m-6-6l6 6-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>

          <div className={styles.step}>
            <div className={styles.stepNumber}>02</div>
            <div className={styles.stepContent}>
              <h3 className={styles.stepTitle}>Build 5-min Windows</h3>
              <p className={styles.stepDescription}>
                Segment the overnight recording into non-overlapping 5-minute windows, 
                computing RR intervals and R-peak amplitudes.
              </p>
              <div className={styles.stepDetails}>
                <div className={styles.detail}>
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                    <path d="M8 3v10M3 8h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                  </svg>
                  <span>RRI time series (channel 1)</span>
                </div>
                <div className={styles.detail}>
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                    <path d="M8 3v10M3 8h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                  </svg>
                  <span>R-peak amplitude (channel 2)</span>
                </div>
              </div>
            </div>
          </div>

          <div className={styles.arrow}>
            <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
              <path d="M10 20h20m-6-6l6 6-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>

          <div className={styles.step}>
            <div className={styles.stepNumber}>03</div>
            <div className={styles.stepContent}>
              <h3 className={styles.stepTitle}>Two-Channel Input</h3>
              <p className={styles.stepDescription}>
                Feed both channels into the hybrid CNN–Transformer model for 
                binary classification (apnea vs. normal).
              </p>
              <div className={styles.stepDetails}>
                <div className={styles.detail}>
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                    <path d="M8 3v10M3 8h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                  </svg>
                  <span>Shape: (batch, 2, sequence_length)</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Architecture;

