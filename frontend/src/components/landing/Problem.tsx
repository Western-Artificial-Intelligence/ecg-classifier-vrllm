import React from 'react';
import styles from '../../styles/Problem.module.css';

const Problem: React.FC = () => {
  return (
    <section id="why" className={styles.problem}>
      <div className={styles.problemContainer}>
        <div className={styles.sectionLabel}>Why This Matters</div>
        <h2 className={styles.title}>Sleep apnea diagnosis is slow and expensive</h2>
        <p className={styles.description}>
          Traditional polysomnography (PSG) requires overnight hospital stays, specialized
          equipment, and expert technicians—resulting in long wait times and high costs.
          Millions remain undiagnosed.
        </p>

        <div className={styles.problemGrid}>
          <div className={styles.problemCard}>
            <div className={styles.problemIcon}>
              <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                <circle cx="16" cy="16" r="12" stroke="currentColor" strokeWidth="2" fill="none" />
                <path d="M16 8v8l4 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
            </div>
            <h3 className={styles.problemTitle}>Months-long wait times</h3>
            <p className={styles.problemText}>
              Patients often wait 3–6 months for a PSG appointment, delaying treatment
              and worsening health outcomes.
            </p>
          </div>

          <div className={styles.problemCard}>
            <div className={styles.problemIcon}>
              <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                <path d="M6 26h20M10 22V10M16 22V6M22 22v-8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
            </div>
            <h3 className={styles.problemTitle}>High cost barrier</h3>
            <p className={styles.problemText}>
              Full PSG studies cost $1,000–$3,000, creating barriers to access
              for underserved populations.
            </p>
          </div>

          <div className={styles.problemCard}>
            <div className={styles.problemIcon}>
              <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                <path d="M16 4v12l8 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                <circle cx="16" cy="16" r="12" stroke="currentColor" strokeWidth="2" fill="none" />
              </svg>
            </div>
            <h3 className={styles.problemTitle}>Scalability limits</h3>
            <p className={styles.problemText}>
              Sleep labs can't scale to meet demand—we need automated, accessible
              screening tools for at-risk populations.
            </p>
          </div>
        </div>

        <div className={styles.solution}>
          <div className={styles.solutionBadge}>Our Approach</div>
          <h3 className={styles.solutionTitle}>
            Screen with single-lead ECG, triage intelligently
          </h3>
          <p className={styles.solutionText}>
            NeuralApnea Triage analyzes affordable overnight ECG recordings to identify
            high-risk patients who need immediate PSG referral—enabling clinicians to
            prioritize resources and reduce diagnostic delays.
          </p>
        </div>
      </div>
    </section>
  );
};

export default Problem;
