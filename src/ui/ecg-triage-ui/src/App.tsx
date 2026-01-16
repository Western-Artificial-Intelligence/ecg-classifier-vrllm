import React, { useState, useEffect } from 'react';
import styles from './styles/App.module.css';
import EcgChart from './components/EcgChart';

function App() {
  const [ecgData, setEcgData] = useState<number[]>([]);
  const [startIndex, setStartIndex] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const viewWindowSize = 500; // Number of data points to display in the chart

  useEffect(() => {
    const fetchEcgData = async () => {
      try {
        setLoading(true);
        setError(null);
        // Assuming backend is running on port 8000
        const response = await fetch('http://localhost:8000/api/ecg_data/a01.dat');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const result = await response.json();
        if (result && result.data) {
          setEcgData(result.data);
        } else {
          throw new Error('Invalid data format received from backend');
        }
      } catch (e: any) {
        setError(e.message);
        console.error("Failed to fetch ECG data:", e);
      } finally {
        setLoading(false);
      }
    };

    fetchEcgData();
  }, []); // Empty dependency array means this runs once on mount

  const handleSliderChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setStartIndex(Number(event.target.value));
  };

  return (
    <div className={styles.appContainer}>
      <div className={styles.topSection}>
        {/* Left column: ECG Display and Patient Info */}
        <div className={styles.leftColumn}>
          {/* ECG Display */}
          <div className={styles.ecgDisplayArea}>
            <h2>ECG Signal Display</h2>
            {loading && <div className={styles.ecgChartPlaceholder}>Loading ECG Data...</div>}
            {error && <div className={styles.ecgChartPlaceholder} style={{ color: 'red' }}>Error: {error}</div>}
            {!loading && !error && ecgData.length > 0 && (
              <div className={styles.ecgChartWrapper}>
                <EcgChart
                  dataPoints={ecgData}
                  viewWindowSize={viewWindowSize}
                  startIndex={startIndex}
                />
              </div>
            )}
            {!loading && !error && ecgData.length === 0 && (
              <div className={styles.ecgChartPlaceholder}>No ECG data available.</div>
            )}

            {/* Scrollable timeline control */}
            <div className={styles.ecgTimelineControl}>
              <input
                type="range"
                min={0}
                max={Math.max(0, ecgData.length - viewWindowSize)}
                value={startIndex}
                onChange={handleSliderChange}
                className={styles.timelineSlider}
                disabled={ecgData.length <= viewWindowSize}
              />
              <p>Viewing samples {startIndex} to {Math.min(startIndex + viewWindowSize, ecgData.length)} of {ecgData.length}</p>
            </div>
          </div>

          {/* Patient Information Box */}
          <div className={styles.patientInfoBox}>
            <h3>Patient Information</h3>
            <div className={styles.patientDetails}>
              <div className={styles.patientInfoRow}>
                <span className={styles.patientLabel}>Patient Name:</span>
                <span className={styles.patientValue}>John Doe</span>
              </div>
              <div className={styles.patientInfoRow}>
                <span className={styles.patientLabel}>Patient #:</span>
                <span className={styles.patientValue}>P-2024-001</span>
              </div>
              <div className={styles.patientInfoRow}>
                <span className={styles.patientLabel}>File:</span>
                <span className={styles.patientValue}>a01.dat</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right column: Chat Window */}
        <div className={styles.chatWindowArea}>
          <h2>AI Assistant</h2>
          <div className={styles.chatMessages}>
            <div className={styles.chatMessage}>
              <div className={styles.messageHeader}>System</div>
              <div className={styles.messageContent}>Welcome to ECG Triage Assistant. How can I help you analyze this ECG?</div>
            </div>
            <div className={styles.chatMessage}>
              <div className={styles.messageHeader}>User</div>
              <div className={styles.messageContent}>What anomalies do you see in this ECG?</div>
            </div>
            <div className={styles.chatMessage}>
              <div className={styles.messageHeader}>Assistant</div>
              <div className={styles.messageContent}>Analyzing the ECG data... This is a placeholder for chat responses.</div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom: Text Prompt Box */}
      <div className={styles.bottomPromptArea}>
        <input
          type="text"
          className={styles.promptInput}
          placeholder="Type your question or command here..."
        />
        <button className={styles.sendButton}>Send</button>
      </div>
    </div>
  );
}

export default App;