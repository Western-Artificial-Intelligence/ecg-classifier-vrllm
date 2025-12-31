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
      <div className={styles.mainContent}>
        {/* Top-left: ECG Display (2/3 width) */}
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

        {/* Top-right: Patient Info & Results (1/3 width) */}
        <div className={styles.infoResultsArea}>
          <h2>Patient Information & Results</h2>
          {/* Placeholder for patient info, upload controls, and ML results */}
          <div className={styles.patientControls}>
            <p>Patient controls, file upload, etc.</p>
          </div>
          <div className={styles.mlResults}>
            <p>ML Classification Results</p>
          </div>
        </div>
      </div>

      {/* Bottom: Potentially more controls or other info */}
      <div className={styles.bottomPanel}>
        <p>Bottom Panel for additional controls or information</p>
      </div>
    </div>
  );
}

export default App;