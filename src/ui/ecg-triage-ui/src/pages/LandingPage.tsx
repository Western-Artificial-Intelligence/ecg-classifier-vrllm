import React from 'react';
import { useNavigate } from 'react-router-dom';
import styles from '../styles/LandingPage.module.css';

const LandingPage: React.FC = () => {
  const navigate = useNavigate();

  const handleGetStarted = () => {
    navigate('/analysis');
  };

  return (
    <div className={styles.landingContainer}>
      {/* Navigation Header */}
      <nav className={styles.navbar}>
        <div className={styles.navContent}>
          <div className={styles.logo}>
            <div className={styles.logoIcon}>
              <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M4 16L8 16L11 8L15 24L19 12L22 16L28 16" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                <circle cx="16" cy="16" r="14" stroke="currentColor" strokeWidth="2" fill="none"/>
              </svg>
            </div>
            <span className={styles.logoText}>ApneaScreen</span>
          </div>
          <div className={styles.navLinks}>
            <a href="#about" className={styles.navLink}>About</a>
            <a href="#features" className={styles.navLink}>Features</a>
            <a href="#how-it-works" className={styles.navLink}>How It Works</a>
            <button onClick={handleGetStarted} className={styles.navButton}>Launch App</button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className={styles.heroSection}>
        <div className={styles.heroContent}>
          <div className={styles.badge}>
            <span className={styles.badgeDot}></span>
            <span>AI-Powered Clinical Assistant</span>
          </div>
          <h1 className={styles.heroTitle}>
            Sleep Apnea Screening
            <br />
            <span className={styles.titleGradient}>Reimagined with AI</span>
          </h1>
          <p className={styles.heroDescription}>
            A clinical screening assistant that analyzes overnight ECG recordings to identify 
            potential sleep apnea events—helping clinicians prioritize cases and make 
            informed decisions faster.
          </p>
          <div className={styles.heroActions}>
            <button onClick={handleGetStarted} className={styles.primaryButton}>
              Start Analysis
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                <path d="M4 10h12M12 6l4 4-4 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              </svg>
            </button>
            <button className={styles.secondaryButton}>
              <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                <path d="M8 5v10l8-5z"/>
              </svg>
              Watch Demo
            </button>
          </div>
          <div className={styles.trustBadges}>
            <div className={styles.trustItem}>
              <span className={styles.trustIcon}>✓</span>
              <span>Clinical-Grade Analysis</span>
            </div>
            <div className={styles.trustItem}>
              <span className={styles.trustIcon}>✓</span>
              <span>Explainable AI</span>
            </div>
            <div className={styles.trustItem}>
              <span className={styles.trustIcon}>✓</span>
              <span>HIPAA-Ready Architecture</span>
            </div>
          </div>
        </div>
        <div className={styles.heroVisual}>
          <div className={styles.visualCard}>
            <div className={styles.cardHeader}>
              <div className={styles.cardTitle}>ECG Analysis</div>
              <div className={styles.cardStatus}>
                <span className={styles.statusDot}></span>
                Processing
              </div>
            </div>
            <div className={styles.mockChart}>
              <svg width="100%" height="120" viewBox="0 0 400 120" preserveAspectRatio="none">
                <defs>
                  <linearGradient id="chartGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor="rgba(99, 102, 241, 0.4)" />
                    <stop offset="100%" stopColor="rgba(99, 102, 241, 0.0)" />
                  </linearGradient>
                </defs>
                <path 
                  d="M0,60 L20,58 L40,55 L45,45 L50,20 L55,45 L60,58 L80,60 L100,58 L120,55 L125,45 L130,20 L135,45 L140,58 L160,60 L180,58 L200,62 L205,52 L210,25 L215,52 L220,62 L240,60 L260,58 L280,55 L285,45 L290,20 L295,45 L300,58 L320,60 L340,58 L360,60 L380,58 L400,60" 
                  stroke="#6366f1" 
                  strokeWidth="2" 
                  fill="url(#chartGradient)"
                />
              </svg>
            </div>
            <div className={styles.cardMetrics}>
              <div className={styles.metric}>
                <span className={styles.metricLabel}>Risk Level</span>
                <span className={styles.metricValue}>High</span>
              </div>
              <div className={styles.metric}>
                <span className={styles.metricLabel}>Confidence</span>
                <span className={styles.metricValue}>94%</span>
              </div>
              <div className={styles.metric}>
                <span className={styles.metricLabel}>Segments</span>
                <span className={styles.metricValue}>12</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className={styles.statsSection}>
        <div className={styles.statsGrid}>
          <div className={styles.statCard}>
            <div className={styles.statNumber}>94%</div>
            <div className={styles.statLabel}>Detection Accuracy</div>
          </div>
          <div className={styles.statCard}>
            <div className={styles.statNumber}>3 min</div>
            <div className={styles.statLabel}>Average Analysis Time</div>
          </div>
          <div className={styles.statCard}>
            <div className={styles.statNumber}>1000+</div>
            <div className={styles.statLabel}>ECG Records Analyzed</div>
          </div>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className={styles.aboutSection}>
        <div className={styles.sectionContent}>
          <div className={styles.sectionBadge}>About the Tool</div>
          <h2 className={styles.sectionTitle}>Clinical Intelligence for Sleep Apnea Detection</h2>
          <p className={styles.sectionDescription}>
            Sleep apnea affects millions globally, but traditional diagnosis requires expensive, 
            time-intensive polysomnography studies. Our AI-powered screening tool analyzes 
            single-lead ECG data to flag high-risk patients, helping clinicians prioritize 
            cases and reduce diagnostic delays.
          </p>
          <div className={styles.featureGrid}>
            <div className={styles.featureCard}>
              <div className={styles.featureIcon}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                  <path d="M12 2L2 7v10c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-10-5z" stroke="currentColor" strokeWidth="2" fill="none"/>
                </svg>
              </div>
              <h3 className={styles.featureTitle}>Not a Medical Device</h3>
              <p className={styles.featureDescription}>
                This is a screening tool, not a diagnostic device. It flags risk and suggests 
                further evaluation via polysomnography.
              </p>
            </div>
            <div className={styles.featureCard}>
              <div className={styles.featureIcon}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                  <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" stroke="currentColor" strokeWidth="2"/>
                </svg>
              </div>
              <h3 className={styles.featureTitle}>Explainable AI</h3>
              <p className={styles.featureDescription}>
                Every prediction is backed by interpretable signals (HRV, entropy changes) 
                so clinicians understand the "why" behind each flag.
              </p>
            </div>
            <div className={styles.featureCard}>
              <div className={styles.featureIcon}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                  <path d="M13 10V3L4 14h7v7l9-11h-7z" stroke="currentColor" strokeWidth="2" fill="none"/>
                </svg>
              </div>
              <h3 className={styles.featureTitle}>Single-Lead ECG</h3>
              <p className={styles.featureDescription}>
                Works with overnight single-lead ECG recordings, making it accessible 
                for remote monitoring and preliminary screening.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className={styles.featuresSection}>
        <div className={styles.sectionContent}>
          <div className={styles.sectionBadge}>Core Features</div>
          <h2 className={styles.sectionTitle}>Everything You Need for Clinical Screening</h2>
          
          <div className={styles.featureShowcase}>
            <div className={styles.showcaseItem}>
              <div className={styles.showcaseNumber}>01</div>
              <div className={styles.showcaseContent}>
                <h3 className={styles.showcaseTitle}>Risk Stratification</h3>
                <p className={styles.showcaseDescription}>
                  Automatically categorizes patients into Low, Medium, or High risk categories 
                  based on ECG analysis, with confidence scores to guide decision-making.
                </p>
              </div>
            </div>
            <div className={styles.showcaseItem}>
              <div className={styles.showcaseNumber}>02</div>
              <div className={styles.showcaseContent}>
                <h3 className={styles.showcaseTitle}>Segment Highlighting</h3>
                <p className={styles.showcaseDescription}>
                  Pinpoints exact time windows in the overnight recording where apnea events 
                  are suspected, allowing targeted review by clinicians.
                </p>
              </div>
            </div>
            <div className={styles.showcaseItem}>
              <div className={styles.showcaseNumber}>03</div>
              <div className={styles.showcaseContent}>
                <h3 className={styles.showcaseTitle}>Interpretable Signals</h3>
                <p className={styles.showcaseDescription}>
                  Shows VLF/HF ratio changes, sample entropy, and other biomarkers that 
                  correlate with apnea events—making AI decisions transparent.
                </p>
              </div>
            </div>
            <div className={styles.showcaseItem}>
              <div className={styles.showcaseNumber}>04</div>
              <div className={styles.showcaseContent}>
                <h3 className={styles.showcaseTitle}>Conversational Assistant</h3>
                <p className={styles.showcaseDescription}>
                  Ask questions about the analysis, get explanations, and receive recommendations 
                  for next steps through an AI-powered clinical assistant.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className={styles.howItWorksSection}>
        <div className={styles.sectionContent}>
          <div className={styles.sectionBadge}>How It Works</div>
          <h2 className={styles.sectionTitle}>Simple Workflow, Powerful Insights</h2>
          
          <div className={styles.workflowSteps}>
            <div className={styles.workflowStep}>
              <div className={styles.stepIcon}>
                <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                  <path d="M16 4v24M8 12l8-8 8 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                </svg>
              </div>
              <div className={styles.stepContent}>
                <h3 className={styles.stepTitle}>1. Upload ECG Data</h3>
                <p className={styles.stepDescription}>
                  Upload overnight single-lead ECG recording in standard .dat format.
                </p>
              </div>
            </div>
            <div className={styles.stepConnector}></div>
            <div className={styles.workflowStep}>
              <div className={styles.stepIcon}>
                <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                  <circle cx="16" cy="16" r="12" stroke="currentColor" strokeWidth="2" fill="none"/>
                  <path d="M16 8v8l4 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                </svg>
              </div>
              <div className={styles.stepContent}>
                <h3 className={styles.stepTitle}>2. AI Analysis</h3>
                <p className={styles.stepDescription}>
                  Our model analyzes heart rate variability patterns to detect potential apnea events.
                </p>
              </div>
            </div>
            <div className={styles.stepConnector}></div>
            <div className={styles.workflowStep}>
              <div className={styles.stepIcon}>
                <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                  <rect x="6" y="6" width="20" height="20" rx="2" stroke="currentColor" strokeWidth="2" fill="none"/>
                  <path d="M12 16l3 3 5-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                </svg>
              </div>
              <div className={styles.stepContent}>
                <h3 className={styles.stepTitle}>3. Review Results</h3>
                <p className={styles.stepDescription}>
                  View risk level, flagged segments, and interpretable explanations in an intuitive interface.
                </p>
              </div>
            </div>
            <div className={styles.stepConnector}></div>
            <div className={styles.workflowStep}>
              <div className={styles.stepIcon}>
                <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                  <path d="M8 16h16M16 8v16" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                </svg>
              </div>
              <div className={styles.stepContent}>
                <h3 className={styles.stepTitle}>4. Take Action</h3>
                <p className={styles.stepDescription}>
                  Use AI recommendations to decide next steps: refer for PSG, monitor, or repeat test.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className={styles.ctaSection}>
        <div className={styles.ctaContent}>
          <h2 className={styles.ctaTitle}>Ready to Transform Sleep Apnea Screening?</h2>
          <p className={styles.ctaDescription}>
            Start analyzing ECG data with AI-powered insights today.
          </p>
          <button onClick={handleGetStarted} className={styles.ctaButton}>
            Launch Application
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path d="M4 10h12M12 6l4 4-4 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </button>
        </div>
      </section>

      {/* Footer */}
      <footer className={styles.footer}>
        <div className={styles.footerContent}>
          <div className={styles.footerLogo}>
            <div className={styles.logoIcon}>
              <svg width="24" height="24" viewBox="0 0 32 32" fill="none">
                <path d="M4 16L8 16L11 8L15 24L19 12L22 16L28 16" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                <circle cx="16" cy="16" r="14" stroke="currentColor" strokeWidth="2" fill="none"/>
              </svg>
            </div>
            <span>ApneaScreen</span>
          </div>
          <div className={styles.footerLinks}>
            <a href="#" className={styles.footerLink}>Documentation</a>
            <a href="#" className={styles.footerLink}>Research</a>
            <a href="#" className={styles.footerLink}>Privacy</a>
            <a href="#" className={styles.footerLink}>Contact</a>
          </div>
          <div className={styles.footerDisclaimer}>
            <p>
              <strong>Medical Disclaimer:</strong> This tool is for screening purposes only and does not 
              diagnose sleep apnea. Always consult with qualified healthcare professionals for medical decisions.
            </p>
            <p className={styles.copyright}>© 2026 ApneaScreen. Built for CUCAI Competition.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;

