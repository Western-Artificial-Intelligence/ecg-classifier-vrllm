import React from 'react';
import { useNavigate } from 'react-router-dom';
import styles from '../styles/LandingPage.module.css';

const LandingPage: React.FC = () => {
  const navigate = useNavigate();

  const handleGetStarted = () => {
    navigate('/dashboard');
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
            <span className={styles.logoText}>NeuralApnea Triage</span>
          </div>
          <div className={styles.navLinks}>
            <a href="#about" className={styles.navLink}>About</a>
            <a href="#features" className={styles.navLink}>Features</a>
            <a href="#how-it-works" className={styles.navLink}>How It Works</a>
            <a href="#research" className={styles.navLink}>Research</a>
            <button onClick={handleGetStarted} className={styles.navButton}>Launch App</button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className={styles.heroSection}>
        <div className={styles.heroContent}>
          <div className={styles.badge}>
            <span className={styles.badgeDot}></span>
            <span>CUCAI Competition Demo</span>
          </div>
          <h1 className={styles.heroTitle}>
            Sleep Apnea Screening
            <br />
            <span className={styles.titleGradient}>Reimagined with AI</span>
          </h1>
          <p className={styles.heroDescription}>
            A demonstration triage and screening tool built for the CUCAI competition, showcasing 
            the potential of AI in medical applications. This prototype analyzes overnight ECG 
            recordings to identify potential sleep apnea events—demonstrating how machine learning 
            could help clinicians prioritize cases in real-world settings.
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
            <div className={styles.statNumber}>80%</div>
            <div className={styles.statLabel}>Sleep Apnea Cases Undiagnosed</div>
          </div>
          <div className={styles.statCard}>
            <div className={styles.statNumber}>8-30 mo</div>
            <div className={styles.statLabel}>Wait Time for PSG in Canada</div>
          </div>
          <div className={styles.statCard}>
            <div className={styles.statNumber}>$1-10K</div>
            <div className={styles.statLabel}>Cost Per Polysomnography Study</div>
          </div>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className={styles.aboutSection}>
        <div className={styles.sectionContent}>
          <div className={styles.sectionBadge}>About the Tool</div>
          <h2 className={styles.sectionTitle}>Clinical Intelligence for Sleep Apnea Detection</h2>
          <p className={styles.sectionDescription}>
            Sleep apnea affects hundreds of millions globally, yet approximately 80% of cases remain 
            undiagnosed. Traditional diagnosis through polysomnography is expensive ($1,000–$10,000 per study), 
            time-intensive, and suffers from severe accessibility issues—with wait times ranging from 8 to 30 months 
            in Canada alone. Our AI-powered screening tool analyzes single-lead ECG data to flag high-risk patients, 
            helping prioritize limited diagnostic resources and reduce delays in care.
          </p>
          <div className={styles.featureGrid}>
            <div className={styles.featureCard}>
              <div className={styles.featureIcon}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                  <path d="M12 2L2 7v10c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-10-5z" stroke="currentColor" strokeWidth="2" fill="none"/>
                </svg>
              </div>
              <h3 className={styles.featureTitle}>Demo Tool - Not a Medical Device</h3>
              <p className={styles.featureDescription}>
                This is a demonstration triage/screening tool built for the CUCAI competition to 
                showcase medical AI potential. Not intended for clinical diagnosis. It demonstrates 
                how ECG-based screening could flag risk and suggest further evaluation via polysomnography.
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
                Every prediction is backed by interpretable physiological signals—heart rate variability (HRV) 
                metrics, sample entropy changes, and attention visualizations—so clinicians understand the "why" 
                behind each risk assessment. Grad-CAM highlights the ECG regions driving predictions.
              </p>
            </div>
            <div className={styles.featureCard}>
              <div className={styles.featureIcon}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                  <path d="M13 10V3L4 14h7v7l9-11h-7z" stroke="currentColor" strokeWidth="2" fill="none"/>
                </svg>
              </div>
              <h3 className={styles.featureTitle}>Software-Only, ECG-Based Screening</h3>
              <p className={styles.featureDescription}>
                Works with overnight single-lead ECG recordings already collected during routine cardiac monitoring. 
                No additional sensors, specialized staff, or new clinical workflows required—unlocking diagnostic 
                information from existing data archives without new collection costs.
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
                <h3 className={styles.showcaseTitle}>AI Architecture</h3>
                <p className={styles.showcaseDescription}>
                  A hybrid CNN-Transformer model captures both local morphological features and long-range 
                  temporal dependencies. The CNN extracts R-wave dynamics and beat-shape patterns, while the 
                  Transformer encoder uses multi-head self-attention to model autonomic fluctuations across 
                  multi-minute windows—detecting gradual apnea-related changes traditional methods miss.
                </p>
              </div>
            </div>
            <div className={styles.showcaseItem}>
              <div className={styles.showcaseNumber}>02</div>
              <div className={styles.showcaseContent}>
                <h3 className={styles.showcaseTitle}>Risk Stratification & Triage</h3>
                <p className={styles.showcaseDescription}>
                  Automatically categorizes patients into Low, Medium, or High risk categories with confidence 
                  scores. This triage capability helps allocate limited polysomnography resources to patients 
                  most likely to benefit, improving diagnostic yield while reducing healthcare costs and wait times.
                </p>
              </div>
            </div>
            <div className={styles.showcaseItem}>
              <div className={styles.showcaseNumber}>03</div>
              <div className={styles.showcaseContent}>
                <h3 className={styles.showcaseTitle}>Segment-Level Analysis</h3>
                <p className={styles.showcaseDescription}>
                  Pinpoints exact time windows in the overnight recording where apnea events are suspected. 
                  The model processes multi-minute ECG segments and aggregates per-segment predictions into 
                  an overall apnea burden score, enabling targeted clinical review of high-risk periods.
                </p>
              </div>
            </div>
            <div className={styles.showcaseItem}>
              <div className={styles.showcaseNumber}>04</div>
              <div className={styles.showcaseContent}>
                <h3 className={styles.showcaseTitle}>Physiological Biomarkers</h3>
                <p className={styles.showcaseDescription}>
                  An explainability agent operates on model outputs and derived metrics—VLF/HF ratio, 
                  sample entropy, R-peak amplitude variability—to highlight apnea-like segments. These 
                  interpretable biomarkers correlate with intermittent hypoxia and autonomic instability, 
                  making AI decisions transparent and clinically meaningful.
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

      {/* Research Section */}
      <section id="research" className={styles.researchSection}>
        <div className={styles.sectionContent}>
          <div className={styles.sectionBadge}>Research</div>
          <h2 className={styles.sectionTitle}>Research Paper</h2>
          <p className={styles.sectionDescription}>
            Our work explores CNN-Transformer architectures for ECG-based sleep apnea detection, 
            combining convolutional feature extraction with attention mechanisms to capture 
            long-range temporal dependencies in heart rate variability patterns.
          </p>
          
          <div className={styles.researchPlaceholder}>
            <div className={styles.placeholderIcon}>
              <svg width="64" height="64" viewBox="0 0 24 24" fill="none">
                <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              </svg>
            </div>
            <h3 className={styles.placeholderTitle}>Full Research Paper Coming Soon</h3>
            <p className={styles.placeholderText}>
              We're currently preparing the complete research paper for publication. 
              The paper will include detailed methodology, architecture design, evaluation metrics, 
              and clinical implications of our CNN-Transformer approach to sleep apnea screening.
            </p>
            <div className={styles.placeholderNote}>
              <strong>Paper Title:</strong> AgenticCardioGram: Machine Learning Powered ECG Analysis System for Sleep Apnea Classification
            </div>
          </div>
        </div>
      </section>

      {/* Team Section */}
      <section id="team" className={styles.teamSection}>
        <div className={styles.sectionContent}>
          <div className={styles.sectionBadge}>Our Team</div>
          <h2 className={styles.sectionTitle}>Meet the Team</h2>
          <p className={styles.sectionDescription}>
            Built by a multidisciplinary team at Western University for the CUCAI competition.
          </p>
          
          <div className={styles.teamGrid}>
            <div className={styles.teamCard}>
              <div className={styles.teamAvatar}>OO</div>
              <h3 className={styles.teamName}>Oliver Olejar</h3>
              <p className={styles.teamAffiliation}>Western University</p>
              <p className={styles.teamEmail}>oolejar@uwo.ca</p>
            </div>
            <div className={styles.teamCard}>
              <div className={styles.teamAvatar}>DK</div>
              <h3 className={styles.teamName}>Daniel Kaminsky</h3>
              <p className={styles.teamAffiliation}>Western University</p>
              <p className={styles.teamEmail}>dkamins7@uwo.ca</p>
            </div>
            <div className={styles.teamCard}>
              <div className={styles.teamAvatar}>AL</div>
              <h3 className={styles.teamName}>Annie Liu</h3>
              <p className={styles.teamAffiliation}>Western University</p>
              <p className={styles.teamEmail}>yliu5349@uwo.ca</p>
            </div>
            <div className={styles.teamCard}>
              <div className={styles.teamAvatar}>JM</div>
              <h3 className={styles.teamName}>John MacPhie</h3>
              <p className={styles.teamAffiliation}>Western University</p>
              <p className={styles.teamEmail}>jmacphi2@uwo.ca</p>
            </div>
            <div className={styles.teamCard}>
              <div className={styles.teamAvatar}>SS</div>
              <h3 className={styles.teamName}>Sneha Shah</h3>
              <p className={styles.teamAffiliation}>Western University</p>
              <p className={styles.teamEmail}>sshah495@uwo.ca</p>
            </div>
            <div className={styles.teamCard}>
              <div className={styles.teamAvatar}>NK</div>
              <h3 className={styles.teamName}>Noah Kostesku</h3>
              <p className={styles.teamAffiliation}>Western University</p>
              <p className={styles.teamEmail}>nkostes@uwo.ca</p>
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
            <span>NeuralApnea Triage</span>
          </div>
          <div className={styles.footerLinks}>
            <a href="#" className={styles.footerLink}>Documentation</a>
            <a href="#" className={styles.footerLink}>Research</a>
            <a href="#" className={styles.footerLink}>Privacy</a>
            <a href="#" className={styles.footerLink}>Contact</a>
          </div>
          <div className={styles.footerDisclaimer}>
            <p>
              <strong>Medical Disclaimer:</strong> NeuralApnea Triage is a demonstration tool built 
              for educational purposes and the CUCAI competition. It is NOT a medical device and is 
              NOT intended for clinical diagnosis or treatment decisions. This system demonstrates 
              the potential of AI-powered ECG screening for sleep apnea triage. Always consult with 
              qualified healthcare professionals for medical decisions and diagnosis.
            </p>
            <p className={styles.copyright}>© 2026 NeuralApnea Triage. Built for CUCAI Competition.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;

