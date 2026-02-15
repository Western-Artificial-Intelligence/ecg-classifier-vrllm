import React, { useState } from 'react';
import styles from '../../styles/Demo.module.css';

interface DemoProps {
  onDemoClick: () => void;
}

const Demo: React.FC<DemoProps> = ({ onDemoClick }) => {
  const [activeTab, setActiveTab] = useState<'ecg' | 'risk' | 'explanation'>('ecg');

  return (
    <section id="demo" className={styles.demo}>
      <div className={styles.demoContainer}>
        <div className={styles.sectionLabel}>Interactive Preview</div>
        <h2 className={styles.title}>See the system in action</h2>
        <p className={styles.subtitle}>
          Explore a sample analysis with realistic ECG data, risk predictions, and
          explainability outputs.
        </p>

        <div className={styles.demoCard}>
          <div className={styles.demoTabs}>
            <button
              className={`${styles.tab} ${activeTab === 'ecg' ? styles.active : ''}`}
              onClick={() => setActiveTab('ecg')}
            >
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                <path d="M3 10h3l2-6 3 12 3-9 2 3h4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              ECG Signal
            </button>
            <button
              className={`${styles.tab} ${activeTab === 'risk' ? styles.active : ''}`}
              onClick={() => setActiveTab('risk')}
            >
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                <path d="M3 17h14M5 13v4M10 7v10M15 10v7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
              Risk Timeline
            </button>
            <button
              className={`${styles.tab} ${activeTab === 'explanation' ? styles.active : ''}`}
              onClick={() => setActiveTab('explanation')}
            >
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                <circle cx="10" cy="10" r="7" stroke="currentColor" strokeWidth="2" fill="none" />
                <path d="M10 7v3M10 13h.01" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
              Explanations
            </button>
          </div>

          <div className={styles.demoContent}>
            {activeTab === 'ecg' && (
              <div className={styles.ecgView}>
                <div className={styles.viewHeader}>
                  <div className={styles.patientInfo}>
                    <span className={styles.patientId}>Patient: a01</span>
                    <span className={styles.duration}>Duration: 8h 23m</span>
                  </div>
                  <div className={styles.viewControls}>
                    <button className={styles.controlBtn}>
                      <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                        <path d="M4 8h8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                      </svg>
                    </button>
                    <button className={styles.controlBtn}>
                      <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                        <path d="M8 4v8M4 8h8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                      </svg>
                    </button>
                  </div>
                </div>
                <svg className={styles.ecgPlot} viewBox="0 0 800 300" preserveAspectRatio="none">
                  <defs>
                    <linearGradient id="ecgGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                      <stop offset="0%" stopColor="rgba(99, 102, 241, 0.2)" />
                      <stop offset="100%" stopColor="rgba(99, 102, 241, 0.0)" />
                    </linearGradient>
                  </defs>
                  <rect x="0" y="0" width="800" height="300" fill="#fafbfc" />
                  <g opacity="0.15">
                    {[...Array(27)].map((_, i) => (
                      <line key={`v${i}`} x1={i * 30} y1="0" x2={i * 30} y2="300" stroke="#6366f1" strokeWidth="1" />
                    ))}
                    {[...Array(10)].map((_, i) => (
                      <line key={`h${i}`} x1="0" y1={i * 30} x2="800" y2={i * 30} stroke="#6366f1" strokeWidth="1" />
                    ))}
                  </g>
                  <path d="M0,150 L40,150 L50,148 L55,145 L58,135 L62,90 L66,135 L70,148 L80,150 L120,150 L130,148 L135,145 L138,135 L142,90 L146,135 L150,148 L160,150 L200,152 L210,156 L215,162 L218,172 L222,200 L226,172 L230,156 L240,152 L280,152 L290,156 L295,162 L298,172 L302,200 L306,172 L310,156 L320,152 L360,150 L370,148 L375,145 L378,135 L382,90 L386,135 L390,148 L400,150 L440,150 L450,148 L455,145 L458,135 L462,90 L466,135 L470,148 L480,150 L520,153 L530,158 L535,165 L538,176 L542,205 L546,176 L550,158 L560,153 L600,152 L610,157 L615,163 L618,174 L622,202 L626,174 L630,157 L640,152 L680,150 L690,148 L695,145 L698,135 L702,90 L706,135 L710,148 L720,150 L760,150 L770,148 L775,145 L778,135 L782,90 L786,135 L790,148 L800,150" stroke="#6366f1" strokeWidth="2.5" fill="url(#ecgGrad)" />
                  <rect x="200" y="0" width="120" height="300" fill="rgba(239, 68, 68, 0.08)" />
                  <rect x="520" y="0" width="140" height="300" fill="rgba(239, 68, 68, 0.08)" />
                </svg>
                <div className={styles.timeMarkers}>
                  <span>00:00</span>
                  <span>01:00</span>
                  <span>02:00</span>
                  <span>03:00</span>
                  <span>04:00</span>
                  <span>05:00</span>
                  <span>06:00</span>
                  <span>07:00</span>
                  <span>08:00</span>
                </div>
              </div>
            )}

            {activeTab === 'risk' && (
              <div className={styles.riskView}>
                <div className={styles.riskHeader}>
                  <div className={styles.overallRisk}>
                    <span className={styles.riskLabel}>Overall Risk Score</span>
                    <div className={styles.riskScore}>
                      <span className={styles.scoreValue}>7.8</span>
                      <span className={styles.scoreMax}>/10</span>
                      <span className={styles.riskBadge + ' ' + styles.high}>High Risk</span>
                    </div>
                  </div>
                  <div className={styles.riskStats}>
                    <div className={styles.riskStat}>
                      <span className={styles.statLabel}>Apnea segments</span>
                      <span className={styles.statValue}>23 / 101</span>
                    </div>
                    <div className={styles.riskStat}>
                      <span className={styles.statLabel}>Avg confidence</span>
                      <span className={styles.statValue}>87%</span>
                    </div>
                  </div>
                </div>
                <div className={styles.riskChart}>
                  <div className={styles.chartLabel}>Per-segment classification over time</div>
                  <div className={styles.barChart}>
                    {[0.2, 0.3, 0.8, 0.9, 0.7, 0.4, 0.2, 0.1, 0.3, 0.6, 0.8, 0.9, 0.5, 0.3, 0.2, 0.7, 0.8, 0.6, 0.3, 0.2].map((height, i) => (
                      <div
                        key={i}
                        className={styles.bar}
                        style={{
                          height: `${height * 100}%`,
                          backgroundColor: height > 0.6 ? '#ef4444' : height > 0.4 ? '#f59e0b' : '#10b981'
                        }}
                      ></div>
                    ))}
                  </div>
                  <div className={styles.chartAxis}>
                    <span>00:00</span>
                    <span>02:00</span>
                    <span>04:00</span>
                    <span>06:00</span>
                    <span>08:00</span>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'explanation' && (
              <div className={styles.explanationView}>
                <div className={styles.explanationCard}>
                  <h4 className={styles.explanationTitle}>Key Findings</h4>
                  <div className={styles.findings}>
                    <div className={styles.finding}>
                      <div className={styles.findingIcon + ' ' + styles.high}>!</div>
                      <div className={styles.findingContent}>
                        <div className={styles.findingLabel}>High VLF/HF Ratio Spike</div>
                        <div className={styles.findingText}>
                          At 02:15–02:35, VLF/HF ratio increased from 1.8 to 4.3, indicating
                          sympathetic activation consistent with apnea arousal.
                        </div>
                      </div>
                    </div>
                    <div className={styles.finding}>
                      <div className={styles.findingIcon + ' ' + styles.medium}>i</div>
                      <div className={styles.findingContent}>
                        <div className={styles.findingLabel}>Reduced Sample Entropy</div>
                        <div className={styles.findingText}>
                          Sample entropy dropped to 0.62 during 04:20–04:40, suggesting
                          reduced HRV complexity typical of apnea events.
                        </div>
                      </div>
                    </div>
                    <div className={styles.finding}>
                      <div className={styles.findingIcon + ' ' + styles.high}>!</div>
                      <div className={styles.findingContent}>
                        <div className={styles.findingLabel}>Prolonged RR Interval Variability</div>
                        <div className={styles.findingText}>
                          RMSSD spiked 78% above baseline at 06:50, indicating sudden autonomic
                          shifts characteristic of post-apnea recovery.
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className={styles.explanationCard}>
                  <h4 className={styles.explanationTitle}>Clinical Recommendation</h4>
                  <div className={styles.recommendation}>
                    <div className={styles.recommendationIcon}>
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                        <path d="M12 2L2 7v10c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-10-5z" stroke="currentColor" strokeWidth="2" fill="none" />
                        <path d="M9 12l2 2 4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    </div>
                    <div className={styles.recommendationContent}>
                      <p>
                        <strong>High-priority referral for polysomnography.</strong> The analysis
                        detected 23 probable apnea events over an 8-hour recording with 87% average
                        confidence. HRV biomarkers strongly correlate with obstructive sleep apnea patterns.
                      </p>
                      <p>
                        Recommend expedited PSG scheduling and consider interim CPAP trial if wait
                        time exceeds 4 weeks.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className={styles.demoFooter}>
            <button className={styles.tryButton} onClick={onDemoClick}>
              Try the Full Web App
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M3 8h10m-4-4l4 4-4 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Demo;
