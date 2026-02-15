import React from 'react';
import styles from '../../styles/Hero.module.css';

interface HeroProps {
  onDemoClick: () => void;
}

const Hero: React.FC<HeroProps> = ({ onDemoClick }) => {
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
          <div className={styles.productCard}>
            <div className={styles.cardHeader}>
              <div className={styles.cardTabs}>
                <div className={styles.cardTab + ' ' + styles.active}>ECG Signal</div>
                <div className={styles.cardTab}>Risk Timeline</div>
                <div className={styles.cardTab}>Explanation</div>
              </div>
              <div className={styles.cardStatus}>
                <div className={styles.statusDot}></div>
                <span>Analyzing</span>
              </div>
            </div>

            <div className={styles.cardBody}>
              <svg className={styles.ecgMock} viewBox="0 0 600 200" preserveAspectRatio="none">
                <defs>
                  <linearGradient id="ecgGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor="rgba(99, 102, 241, 0.3)" />
                    <stop offset="100%" stopColor="rgba(99, 102, 241, 0.0)" />
                  </linearGradient>
                  <linearGradient id="highlightGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor="rgba(239, 68, 68, 0.2)" />
                    <stop offset="100%" stopColor="rgba(239, 68, 68, 0.0)" />
                  </linearGradient>
                </defs>

                {/* Grid */}
                <g opacity="0.1">
                  {[...Array(20)].map((_, i) => (
                    <line key={`v${i}`} x1={i * 30} y1="0" x2={i * 30} y2="200" stroke="#6366f1" strokeWidth="0.5" />
                  ))}
                  {[...Array(8)].map((_, i) => (
                    <line key={`h${i}`} x1="0" y1={i * 25} x2="600" y2={i * 25} stroke="#6366f1" strokeWidth="0.5" />
                  ))}
                </g>

                {/* Highlight zones (apnea regions) */}
                <rect x="180" y="0" width="80" height="200" fill="url(#highlightGradient)" />
                <rect x="420" y="0" width="100" height="200" fill="url(#highlightGradient)" />

                {/* ECG waveform */}
                <path
                  d="M0,100 L30,100 L40,98 L45,95 L48,85 L52,40 L56,85 L60,98 L70,100 
                     L100,100 L110,98 L115,95 L118,85 L122,40 L126,85 L130,98 L140,100 
                     L170,102 L180,105 L185,108 L188,115 L192,140 L196,115 L200,105 L210,102 
                     L240,102 L250,105 L255,108 L258,115 L262,140 L266,115 L270,105 L280,102 
                     L310,100 L320,98 L325,95 L328,85 L332,40 L336,85 L340,98 L350,100 
                     L380,100 L390,98 L395,95 L398,85 L402,40 L406,85 L410,98 L420,100 
                     L450,103 L460,107 L465,112 L468,120 L472,145 L476,120 L480,107 L490,103 
                     L520,102 L530,106 L535,110 L538,118 L542,142 L546,118 L550,106 L560,102 
                     L590,100 L600,100"
                  stroke="#6366f1"
                  strokeWidth="2"
                  fill="url(#ecgGradient)"
                />
              </svg>

              <div className={styles.riskTimeline}>
                <div className={styles.timelineBar}>
                  <div className={styles.timelineSegment + ' ' + styles.low} style={{ width: '30%' }}></div>
                  <div className={styles.timelineSegment + ' ' + styles.high} style={{ width: '13%' }}></div>
                  <div className={styles.timelineSegment + ' ' + styles.medium} style={{ width: '24%' }}></div>
                  <div className={styles.timelineSegment + ' ' + styles.high} style={{ width: '17%' }}></div>
                  <div className={styles.timelineSegment + ' ' + styles.low} style={{ width: '16%' }}></div>
                </div>
                <div className={styles.timelineLabels}>
                  <span>00:00</span>
                  <span>02:00</span>
                  <span>04:00</span>
                  <span>06:00</span>
                  <span>08:00</span>
                </div>
              </div>
            </div>

            <div className={styles.cardFooter}>
              <div className={styles.metric}>
                <span className={styles.metricLabel}>Overall Risk</span>
                <span className={styles.metricValue + ' ' + styles.high}>High</span>
              </div>
              <div className={styles.metric}>
                <span className={styles.metricLabel}>Confidence</span>
                <span className={styles.metricValue}>87%</span>
              </div>
              <div className={styles.metric}>
                <span className={styles.metricLabel}>Apnea Windows</span>
                <span className={styles.metricValue}>23</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
