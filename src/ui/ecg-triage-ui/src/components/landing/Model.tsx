import React from 'react';
import styles from '../../styles/Model.module.css';

const Model: React.FC = () => {
  return (
    <section id="model" className={styles.model}>
      <div className={styles.modelContainer}>
        <div className={styles.sectionLabel}>Model Architecture</div>
        <h2 className={styles.title}>Hybrid CNN–Transformer for sleep apnea detection</h2>
        <p className={styles.subtitle}>
          Combines convolutional feature extraction with self-attention to capture both 
          local HRV patterns and long-range temporal dependencies.
        </p>
        
        <div className={styles.modelDiagram}>
          <div className={styles.layer}>
            <div className={styles.layerBox}>
              <div className={styles.layerTitle}>CNN Feature Extractor</div>
              <div className={styles.layerDetails}>
                <span>1D Conv layers</span>
                <span>Extract local HRV patterns</span>
              </div>
            </div>
          </div>
          
          <div className={styles.layerArrow}>→</div>
          
          <div className={styles.layer}>
            <div className={styles.layerBox}>
              <div className={styles.layerTitle}>Transformer Encoder</div>
              <div className={styles.layerDetails}>
                <span>Multi-head self-attention</span>
                <span>Capture temporal dependencies</span>
              </div>
            </div>
          </div>
          
          <div className={styles.layerArrow}>→</div>
          
          <div className={styles.layer}>
            <div className={styles.layerBox}>
              <div className={styles.layerTitle}>Classification Head</div>
              <div className={styles.layerDetails}>
                <span>Binary output</span>
                <span>Apnea vs. Normal</span>
              </div>
            </div>
          </div>
        </div>

        <div className={styles.metrics}>
          <h3 className={styles.metricsTitle}>Performance Metrics</h3>
          <div className={styles.metricsGrid}>
            <div className={styles.metricCard}>
              <div className={styles.metricValue}>87.3%</div>
              <div className={styles.metricLabel}>Accuracy</div>
              <div className={styles.metricDescription}>Per-segment classification</div>
            </div>
            <div className={styles.metricCard}>
              <div className={styles.metricValue}>85.1%</div>
              <div className={styles.metricLabel}>Sensitivity</div>
              <div className={styles.metricDescription}>True positive rate</div>
            </div>
            <div className={styles.metricCard}>
              <div className={styles.metricValue}>89.4%</div>
              <div className={styles.metricLabel}>Specificity</div>
              <div className={styles.metricDescription}>True negative rate</div>
            </div>
            <div className={styles.metricCard}>
              <div className={styles.metricValue}>0.92</div>
              <div className={styles.metricLabel}>AUC-ROC</div>
              <div className={styles.metricDescription}>Overall discrimination</div>
            </div>
          </div>
        </div>

        <div className={styles.highlights}>
          <div className={styles.highlight}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" fill="none"/>
              <path d="M9 12l2 2 4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            <div>
              <div className={styles.highlightTitle}>Patient-level evaluation</div>
              <div className={styles.highlightText}>Aggregates 5-min predictions to per-patient apnea risk scores</div>
            </div>
          </div>
          <div className={styles.highlight}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" fill="none"/>
              <path d="M9 12l2 2 4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            <div>
              <div className={styles.highlightTitle}>Cross-validation</div>
              <div className={styles.highlightText}>5-fold CV on PhysioNet dataset (70 patients, ~8 hours each)</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Model;

