import React from 'react';
import styles from '../../styles/Footer.module.css';

const Footer: React.FC = () => {
  return (
    <footer id="team" className={styles.footer}>
      <div className={styles.footerContainer}>
        <div className={styles.footerTop}>
          <div className={styles.footerBrand}>
            <div className={styles.logo}>
              <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                <path d="M4 16h4l3-8 4 16 4-12 3 4h6" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
                <circle cx="16" cy="16" r="14" stroke="currentColor" strokeWidth="2" fill="none" />
              </svg>
              <span>NeuralApnea Triage</span>
            </div>
            <p className={styles.tagline}>
              Machine learning-powered sleep apnea screening from single-lead ECG
            </p>
          </div>

          <div className={styles.footerLinks}>
            <div className={styles.linkColumn}>
              <h4 className={styles.linkTitle}>Research</h4>
              <a href="#research" className={styles.link}>Read the Paper</a>
              <a href="#architecture" className={styles.link}>Architecture</a>
              <a href="#model" className={styles.link}>Model Details</a>
              <a href="#explainability" className={styles.link}>Explainability</a>
            </div>
            <div className={styles.linkColumn}>
              <h4 className={styles.linkTitle}>Resources</h4>
              <a href="https://github.com" className={styles.link}>GitHub Repository</a>
              <a href="#demo" className={styles.link}>Interactive Demo</a>
              <a href="#faq" className={styles.link}>FAQ</a>
            </div>
            <div className={styles.linkColumn}>
              <h4 className={styles.linkTitle}>Team</h4>
              <a href="https://uwo.ca" className={styles.link}>Western University</a>
              <a href="https://uwo.ca" className={styles.link}>Western AI</a>
              <a href="mailto:contact@example.com" className={styles.link}>Contact Us</a>
            </div>
          </div>
        </div>

        <div className={styles.footerDivider}></div>

        <div className={styles.footerBottom}>
          <div className={styles.disclaimer}>
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <circle cx="10" cy="10" r="8" stroke="currentColor" strokeWidth="2" fill="none" />
              <path d="M10 6v4M10 13h.01" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
            <p>
              <strong>Medical Disclaimer:</strong> This is a research prototype for educational
              and screening purposes only. It does not diagnose sleep apnea and should not be
              used as a substitute for professional medical advice, diagnosis, or treatment.
              Always consult qualified healthcare professionals for medical decisions.
            </p>
          </div>

          <div className={styles.footerMeta}>
            <p className={styles.copyright}>
              © 2026 NeuralApnea Triage. Built at Western AI / Western University.
            </p>
            <div className={styles.socialLinks}>
              <a href="https://github.com" className={styles.socialLink} aria-label="GitHub">
                <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                  <path d="M10 0C4.477 0 0 4.477 0 10c0 4.42 2.865 8.17 6.839 9.49.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.603-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.463-1.11-1.463-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.831.092-.646.35-1.086.636-1.336-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0110 4.836c.85.004 1.705.114 2.504.336 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C17.137 18.165 20 14.418 20 10c0-5.523-4.477-10-10-10z" />
                </svg>
              </a>
              <a href="https://twitter.com" className={styles.socialLink} aria-label="Twitter">
                <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                  <path d="M6.29 18.251c7.547 0 11.675-6.253 11.675-11.675 0-.178 0-.355-.012-.53A8.348 8.348 0 0020 3.92a8.19 8.19 0 01-2.357.646 4.118 4.118 0 001.804-2.27 8.224 8.224 0 01-2.605.996 4.107 4.107 0 00-6.993 3.743 11.65 11.65 0 01-8.457-4.287 4.106 4.106 0 001.27 5.477A4.073 4.073 0 01.8 7.713v.052a4.105 4.105 0 003.292 4.022 4.095 4.095 0 01-1.853.07 4.108 4.108 0 003.834 2.85A8.233 8.233 0 010 16.407a11.616 11.616 0 006.29 1.84" />
                </svg>
              </a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
