import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import styles from '../styles/NewLandingPage.module.css';
import Hero from '../components/landing/Hero';
import Stats from '../components/landing/Stats';
import Problem from '../components/landing/Problem';
import Architecture from '../components/landing/Architecture';
import Model from '../components/landing/Model';
import Explainability from '../components/landing/Explainability';
import Demo from '../components/landing/Demo';
import FAQ from '../components/landing/FAQ';
import Footer from '../components/landing/Footer';

const NewLandingPage: React.FC = () => {
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
    navigate('/dashboard');
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
              <path d="M4 16h4l3-8 4 16 4-12 3 4h6" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
              <circle cx="16" cy="16" r="14" stroke="currentColor" strokeWidth="2" fill="none"/>
            </svg>
            <span>AgenticCardioGram</span>
          </div>
          <div className={styles.navLinks}>
            <a href="#how-it-works">How it works</a>
            <a href="#research">Research</a>
            <a href="#team">Team</a>
            <a href="https://github.com" target="_blank" rel="noopener noreferrer">GitHub</a>
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

export default NewLandingPage;

