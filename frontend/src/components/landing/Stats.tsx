import React from 'react';
import styles from '../../styles/Stats.module.css';

const Stats: React.FC = () => {
  return (
    <section id="stats" className={styles.stats}>
      <div className={styles.statsContainer}>
        <div className={styles.badge}>Built at Western AI / Western University</div>
        <p className={styles.dataset}>Trained & validated on PhysioNet Apnea-ECG Dataset</p>
        <div className={styles.statsGrid}>
          <div className={styles.statCard}>
            <div className={styles.statValue}>87.3%</div>
            <div className={styles.statLabel}>Per-segment accuracy</div>
          </div>
          <div className={styles.statCard}>
            <div className={styles.statValue}>5-min</div>
            <div className={styles.statLabel}>Window resolution</div>
          </div>
          <div className={styles.statCard}>
            <div className={styles.statValue}>2-channel</div>
            <div className={styles.statLabel}>Time-series input</div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Stats;

