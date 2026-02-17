import React from 'react';
import { useNavigate } from 'react-router-dom';
import styles from '../styles/MainMenu.module.css';

const MainMenu: React.FC = () => {
  const navigate = useNavigate();

  return (
    <main className={styles.mainMenuPage}>
      <header className={styles.topBar}>
        <div className={styles.brand}>
          <span className={styles.brandDot}></span>
          <span>NeuralApnea</span>
        </div>
      </header>

      <div className={styles.bgEffects} aria-hidden="true">
        <div className={styles.blurBlob1}></div>
        <div className={styles.blurBlob2}></div>
      </div>

      <section className={styles.menuCard} aria-label="Main Menu">
        <h1 className={styles.title}>NeuralApnea Triage</h1>
        <p className={styles.subtitle}>Sleep ECG triage workspace</p>
        <div className={styles.actions}>
          <button
            type="button"
            className={styles.secondaryButton}
            onClick={() => navigate('/')}
          >
            Back to Landing Page
          </button>
          <button
            type="button"
            className={styles.primaryButton}
            onClick={() => navigate('/patients')}
          >
            Select Patient
          </button>
        </div>
      </section>
    </main>
  );
};

export default MainMenu;
