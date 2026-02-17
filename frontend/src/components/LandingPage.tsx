import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import styles from '../styles/NewLandingPage.module.css';

// Modular Components
import Hero from './landing/Hero';
import Stats from './landing/Stats';
import Problem from './landing/Problem';
import Architecture from './landing/Architecture';
import Model from './landing/Model';
import Explainability from './landing/Explainability';
import Demo from './landing/Demo';
import FAQ from './landing/FAQ';
import Footer from './landing/Footer';

const LandingPage: React.FC = () => {
  const navigate = useNavigate();
  const [isNavScrolled, setIsNavScrolled] = useState(false);

  React.useEffect(() => {
    const handleScroll = () => {
      setIsNavScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleDemoClick = () => {
    navigate('/menu');
  };

  return (
    <div className={styles.landingPage}>
      {/* Background Effects */}
      <div className={styles.bgEffects}>
        <div className={styles.blurBlob1}></div>
        <div className={styles.blurBlob2}></div>
        <div className={styles.blurBlob3}></div>
      </div>

      {/* Navigation */}
      <nav className={`${styles.navbar} ${isNavScrolled ? styles.navbarScrolled : ''}`}>
        <div className={styles.navContent}>
          <div className={styles.navLogo}>
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
              <path d="M4 16h4l3-8 4 16 4-12 3 4h6" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
              <circle cx="16" cy="16" r="14" stroke="currentColor" strokeWidth="2" fill="none" />
            </svg>
            <span>NeuralApnea Triage</span>
          </div>
          <div className={styles.navLinks}>
            <a href="#why">Why</a>
            <a href="#architecture">Architecture</a>
            <a href="#how-it-works">How it works</a>
            <a href="#research">Research</a>
            <a href="#team">Team</a>
            <button onClick={handleDemoClick} className={styles.navButton}>
              Open Portal
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <Hero onDemoClick={handleDemoClick} />

      {/* Stats */}
      <Stats />

      {/* Problem Section */}
      <Problem />

      {/* Architecture */}
      <Architecture />

      {/* Model Details */}
      <Model />

      {/* Explainability */}
      <Explainability />

      {/* Interactive Demo */}
      <Demo onDemoClick={handleDemoClick} />

      {/* FAQ */}
      <FAQ />

      {/* Footer */}
      <Footer />
    </div>
  );
};

export default LandingPage;
