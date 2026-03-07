import React from 'react';
import styles from '../../styles/Hero.module.css';

interface HeroProps {
  onDemoClick: () => void;
  /** URL path to the hero image. Use a path from public/ (e.g. "/images/hero-visual.png") or pass an imported image URL. */
  heroImageSrc?: string;
  heroImageSrc2?: string;
}

const Hero: React.FC<HeroProps> = ({ onDemoClick, heroImageSrc = '/images/gradcam overlay.png', heroImageSrc2 = '/images/normal apnea overlay.png' }) => {
  return (
    <section className={styles.hero}>
      <div className={styles.heroContainer}>
        <div className={styles.heroText}>
          <h1 className={styles.headline}>
            Sleep Apnea Screening<br />
            From Single-Lead ECG
          </h1>
          <p className={styles.subheadline}>
            Machine learning-powered analysis with explainable AI. Screen for sleep apnea
            using overnight ECG—no expensive polysomnography required for initial triage.
          </p>
          <div className={styles.ctaButtons}>
            <button className={styles.primaryCta} onClick={onDemoClick}>
              <span>View Demo</span>
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                <path d="M4 10h12m-4-4l4 4-4 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
            <a href="#research" className={styles.secondaryCta}>
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                <path d="M9 12h6m-6 4h6M9 8h6m-9 4h.01M6 12h.01M6 16h.01M6 8h.01" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              <span>Read the Paper</span>
            </a>
          </div>
          <div className={styles.badges}>
            <div className={styles.badge}>
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M8 1L2 4v5c0 3.7 2.56 7.16 6 8 3.44-.84 6-4.3 6-8V4l-6-3z" stroke="currentColor" strokeWidth="1.5" fill="none" />
              </svg>
              <span>Single-lead ECG</span>
            </div>
            <div className={styles.badge}>
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <circle cx="8" cy="8" r="6" stroke="currentColor" strokeWidth="1.5" fill="none" />
                <circle cx="8" cy="8" r="2" fill="currentColor" />
              </svg>
              <span>CNN–Transformer</span>
            </div>
            <div className={styles.badge}>
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M8 2v12M2 8h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              </svg>
              <span>Explainability Agent</span>
            </div>
          </div>
        </div>

        <div className={styles.heroVisual}>
          <img
            src={heroImageSrc}
            alt="ECG sleep apnea screening dashboard"
            className={styles.heroImage}
          />
          <img
            src={heroImageSrc2}
            alt="ECG sleep apnea screening dashboard"
            className={styles.heroImage}
          />
        </div>
        
      </div>
    </section>
  );
};

export default Hero;
