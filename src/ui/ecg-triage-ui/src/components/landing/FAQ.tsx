import React, { useState } from 'react';
import styles from '../../styles/FAQ.module.css';

interface FAQItemProps {
  question: string;
  answer: string;
  isOpen: boolean;
  onClick: () => void;
}

const FAQItem: React.FC<FAQItemProps> = ({ question, answer, isOpen, onClick }) => {
  return (
    <div className={`${styles.faqItem} ${isOpen ? styles.open : ''}`}>
      <button className={styles.faqQuestion} onClick={onClick}>
        <span>{question}</span>
        <svg 
          width="20" 
          height="20" 
          viewBox="0 0 20 20" 
          fill="none"
          className={styles.chevron}
        >
          <path d="M6 8l4 4 4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </button>
      <div className={styles.faqAnswer}>
        <p>{answer}</p>
      </div>
    </div>
  );
};

const FAQ: React.FC = () => {
  const [openIndex, setOpenIndex] = useState<number | null>(0);

  const faqs = [
    {
      question: 'Is this a replacement for polysomnography (PSG)?',
      answer: 'No. AgenticCardioGram is a screening and triage tool, not a diagnostic device. It helps clinicians prioritize high-risk patients for PSG referral and reduce diagnostic delays. A positive result should always be confirmed with full polysomnography.'
    },
    {
      question: 'What kind of ECG data does it accept?',
      answer: 'The system works with single-lead overnight ECG recordings, typically 6 to 8 hours in duration. It accepts standard PhysioNet-compatible .dat/.hea formats, though the web interface can adapt to common clinical ECG exports.'
    },
    {
      question: 'How accurate is the model?',
      answer: 'On the PhysioNet Apnea-ECG dataset, the model achieves 87.3% per-segment accuracy, 85.1% sensitivity, 89.4% specificity, and 0.92 AUC-ROC. Performance may vary on unseen clinical populations and should be validated in deployment settings.'
    },
    {
      question: 'Can I use this in a clinical setting?',
      answer: 'This is a research prototype built for academic exploration and has not been cleared by regulatory bodies (FDA, Health Canada, etc.). It should not be used for clinical decision-making without proper validation, institutional review, and regulatory approval.'
    },
    {
      question: 'What are the system requirements?',
      answer: 'The web interface runs in any modern browser. For backend inference, you will need Python 3.9+, PyTorch, and standard scientific libraries (NumPy, SciPy). The model can run on CPU but benefits from GPU acceleration for batch processing.'
    },
    {
      question: 'How does the explainability agent work?',
      answer: 'The explainability module extracts clinically relevant HRV features (VLF/HF ratio, sample entropy, RMSSD) and uses attention-weighted heatmaps (similar to Grad-CAM) to highlight time windows where the model detected apnea-like patterns. These outputs are designed to be interpretable by clinicians familiar with HRV physiology.'
    }
  ];

  return (
    <section id="faq" className={styles.faq}>
      <div className={styles.faqContainer}>
        <div className={styles.sectionLabel}>Frequently Asked Questions</div>
        <h2 className={styles.title}>Everything you need to know</h2>
        
        <div className={styles.faqList}>
          {faqs.map((faq, index) => (
            <FAQItem
              key={index}
              question={faq.question}
              answer={faq.answer}
              isOpen={openIndex === index}
              onClick={() => setOpenIndex(openIndex === index ? null : index)}
            />
          ))}
        </div>
      </div>
    </section>
  );
};

export default FAQ;
