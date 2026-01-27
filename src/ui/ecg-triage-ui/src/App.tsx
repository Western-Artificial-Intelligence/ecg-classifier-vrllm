import React, { useState, useEffect, useRef } from 'react';
import styles from './styles/App.module.css';
import EcgChart from './components/EcgChart';
import MinimapView from './components/MinimapView';
import SummaryChartView from './components/SummaryChartView';

interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface PatientFile {
  name: string;
  folder: string;
  date: Date;
  file?: File; // The actual file object if uploaded
}

interface Prediction {
  minute: number;
  probability: number;
}

interface GradCAMData {
  minute: number;
  imageUrl: string;
  probability: number;
  predictedClass: string;
}

type ZoomLevel = 'DETAIL' | 'MINUTE_1' | 'MINUTE_5' | 'FULL';
type ViewMode = 'waveform' | 'minimap' | 'summary';

const ZOOM_PRESETS = {
  DETAIL: { label: '1x (Detail)', samples: 500 },
  MINUTE_1: { label: '5x (1 min)', samples: 6000 },
  MINUTE_5: { label: '10x (5 min)', samples: 30000 },
  FULL: { label: 'Full Record', samples: -1 } // -1 means all samples
};

function App() {
  const [ecgData, setEcgData] = useState<number[]>([]);
  const [startIndex, setStartIndex] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [isPanelsCollapsed, setIsPanelsCollapsed] = useState<boolean>(false);
  const [openFolders, setOpenFolders] = useState<Set<string>>(new Set(['ECG Recordings']));
  const [activeFile, setActiveFile] = useState<string>('a01.dat');
  const [chatInput, setChatInput] = useState<string>('');
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      role: 'system',
      content: 'Welcome to ECG Triage Assistant. How can I help you analyze this ECG?',
      timestamp: new Date()
    }
  ]);
  const [patientFiles, setPatientFiles] = useState<PatientFile[]>([
    { name: 'a01.dat', folder: 'ECG Recordings', date: new Date('2024-01-10') },
    { name: 'a02.dat', folder: 'ECG Recordings', date: new Date('2024-01-08') },
    { name: 'a03.dat', folder: 'ECG Recordings', date: new Date('2024-01-05') },
    { name: 'test_12_2023.dat', folder: 'Previous Tests', date: new Date('2023-12-15') }
  ]);

  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [predicting, setPredicting] = useState<boolean>(false);
  const [selectedMinute, setSelectedMinute] = useState<number | null>(null);
  const [gradcamData, setGradcamData] = useState<GradCAMData | null>(null);
  const [loadingGradcam, setLoadingGradcam] = useState<boolean>(false);
  const [gradcamCache, setGradcamCache] = useState<Map<number, GradCAMData>>(new Map());
  const [currentZoom, setCurrentZoom] = useState<ZoomLevel>('DETAIL');
  const [viewMode, setViewMode] = useState<ViewMode>('waveform');

  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatMessagesEndRef = useRef<HTMLDivElement>(null);
  
  // Calculate view window size based on current zoom level
  const viewWindowSize = ZOOM_PRESETS[currentZoom].samples === -1 
    ? ecgData.length 
    : ZOOM_PRESETS[currentZoom].samples;

  const toggleFolder = (folderName: string) => {
    setOpenFolders(prev => {
      const newSet = new Set(prev);
      if (newSet.has(folderName)) {
        newSet.delete(folderName);
      } else {
        newSet.add(folderName);
      }
      return newSet;
    });
  };

  // Scroll to bottom of chat when new messages arrive
  useEffect(() => {
    chatMessagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  useEffect(() => {
    const fetchEcgData = async () => {
      try {
        setLoading(true);
        setError(null);
        // Assuming backend is running on port 8000
        const response = await fetch(`http://localhost:8000/api/ecg_data/${activeFile}`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const result = await response.json();
        if (result && result.data) {
          setEcgData(result.data);
          setStartIndex(0); // Reset view when loading new file
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
  }, [activeFile]); // Re-fetch when active file changes

  const handleSliderChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setStartIndex(Number(event.target.value));
  };

  const handleFileSelect = (fileName: string) => {
    setActiveFile(fileName);
  };

  const validateDatFile = (file: File): boolean => {
    return file.name.toLowerCase().endsWith('.dat');
  };

  const handleFileAdd = (files: FileList | null) => {
    if (!files || files.length === 0) return;

    const validFiles: PatientFile[] = [];
    const invalidFiles: string[] = [];

    Array.from(files).forEach(file => {
      if (validateDatFile(file)) {
        validFiles.push({
          name: file.name,
          folder: 'ECG Recordings',
          date: new Date(),
          file: file
        });
      } else {
        invalidFiles.push(file.name);
      }
    });

    if (validFiles.length > 0) {
      setPatientFiles(prev => [...validFiles, ...prev]);
      // TODO: Upload files to backend here
      console.log('Files to upload:', validFiles);
    }

    if (invalidFiles.length > 0) {
      alert(`The following files were rejected (only .dat files allowed):\n${invalidFiles.join('\n')}`);
    }
  };

  const handleAddFileClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    handleFileAdd(event.target.files);
    // Reset input so same file can be selected again
    event.target.value = '';
  };

  const handleFileDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    handleFileAdd(event.dataTransfer.files);
  };

  const handleSendMessage = async () => {
    if (chatInput.trim() === '') return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: chatInput,
      timestamp: new Date()
    };

    setChatMessages(prev => [...prev, userMessage]);
    setChatInput('');

    // TODO: Send message to backend AI agent
    // For now, simulate a response
    try {
      // Simulate API call delay
      setTimeout(() => {
        const assistantMessage: ChatMessage = {
          role: 'assistant',
          content: `I received your message: "${userMessage.content}". This is a placeholder response. Integration with the AI agent is pending.`,
          timestamp: new Date()
        };
        setChatMessages(prev => [...prev, assistantMessage]);
      }, 1000);
    } catch (e: any) {
      console.error('Error sending message:', e);
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  // Fetch predictions for the active file
  const fetchPredictions = async (filename: string) => {
    setPredicting(true);
    try {
      const recordName = filename.replace('.dat', '');
      const response = await fetch(`http://localhost:8000/api/predict/${recordName}`, {
        method: 'POST'
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      setPredictions(result.predictions || []);
    } catch (e: any) {
      console.error('Error fetching predictions:', e);
      setError(`Prediction error: ${e.message}`);
    } finally {
      setPredicting(false);
    }
  };

  // Fetch Grad-CAM visualization for a specific minute
  const fetchGradcam = async (filename: string, minute: number) => {
    // Check cache first
    if (gradcamCache.has(minute)) {
      return gradcamCache.get(minute)!;
    }

    setLoadingGradcam(true);
    try {
      const recordName = filename.replace('.dat', '');
      const response = await fetch(
        `http://localhost:8000/api/gradcam/${recordName}?minute=${minute}`,
        { method: 'POST' }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      const gradcamData: GradCAMData = {
        minute: result.minute,
        imageUrl: result.image_url,
        probability: result.probability,
        predictedClass: result.predicted_class
      };
      
      // Cache the result
      setGradcamCache(prev => new Map(prev).set(minute, gradcamData));
      
      return gradcamData;
    } catch (e: any) {
      console.error('Error fetching Grad-CAM:', e);
      throw e;
    } finally {
      setLoadingGradcam(false);
    }
  };

  // Handle segment click to show Grad-CAM
  const handleSegmentClick = async (minute: number) => {
    setSelectedMinute(minute);
    setGradcamData(null); // Clear previous data
    setLoadingGradcam(true);
    
    try {
      const gradcam = await fetchGradcam(activeFile, minute);
      setGradcamData(gradcam);
    } catch (e) {
      console.error('Failed to load Grad-CAM:', e);
      setLoadingGradcam(false);
    }
  };

  // Generate all Grad-CAMs for apneic minutes
  const handleGenerateAllGradcams = async () => {
    const apneicMinutes = predictions
      .filter(p => p.probability >= 0.5)
      .map(p => p.minute);
    
    for (let i = 0; i < apneicMinutes.length; i++) {
      const minute = apneicMinutes[i];
      if (!gradcamCache.has(minute)) {
        try {
          await fetchGradcam(activeFile, minute);
        } catch (e) {
          console.error(`Failed to generate Grad-CAM for minute ${minute}:`, e);
        }
      }
    }
  };

  // Fetch predictions when activeFile changes
  useEffect(() => {
    if (activeFile) {
      fetchPredictions(activeFile);
    }
  }, [activeFile]);

  // Keyboard shortcuts for zoom and view modes
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // Only trigger if not typing in an input field
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      switch (e.key) {
        case '1':
          setCurrentZoom('DETAIL');
          break;
        case '5':
          setCurrentZoom('MINUTE_1');
          break;
        case '0':
          setCurrentZoom('MINUTE_5');
          break;
        case 'f':
        case 'F':
          setCurrentZoom('FULL');
          break;
        case 'w':
        case 'W':
          setViewMode('waveform');
          break;
        case 'm':
        case 'M':
          setViewMode('minimap');
          break;
        case 's':
        case 'S':
          setViewMode('summary');
          break;
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, []);

  return (
    <div className={styles.appContainer}>
      <div className={styles.mainLayout}>
        {/* Left sidebar: File Selection */}
        {!isPanelsCollapsed && (
          <div className={styles.fileSelectionArea}>
            <button className={styles.backButton}>← Back to Patients</button>
            <h3>Patient: John Doe</h3>
            <button className={styles.addFileButton} onClick={handleAddFileClick}>
              + Add New File
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".dat"
              multiple
              style={{ display: 'none' }}
              onChange={handleFileInputChange}
            />
            
            <div 
              className={styles.fileDropZone}
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleFileDrop}
            >
              <div className={styles.dropZoneText}>
                <span>📁</span>
                <p>Drag & Drop .dat files here</p>
              </div>
            </div>

            <div className={styles.fileTree}>
              {['ECG Recordings', 'Previous Tests'].map(folderName => {
                const filesInFolder = patientFiles.filter(f => f.folder === folderName);
                if (filesInFolder.length === 0) return null;

                return (
                  <React.Fragment key={folderName}>
                    <div className={styles.folderItem} onClick={() => toggleFolder(folderName)}>
                      <span className={styles.folderIcon}>
                        {openFolders.has(folderName) ? '📂' : '📁'}
                      </span>
                      <span className={styles.folderName}>{folderName}</span>
                      <span className={styles.folderToggle}>
                        {openFolders.has(folderName) ? '▼' : '▶'}
                      </span>
                    </div>
                    {openFolders.has(folderName) && (
                      <div className={styles.fileList}>
                        {filesInFolder.map((file) => (
                          <div 
                            key={file.name}
                            className={`${styles.fileItem} ${activeFile === file.name ? styles.activeFile : ''}`}
                            onClick={() => handleFileSelect(file.name)}
                          >
                            <span className={styles.fileIcon}>📄</span>
                            <span className={styles.fileName}>{file.name}</span>
                            <span className={styles.fileDate}>
                              {file.date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </React.Fragment>
                );
              })}
            </div>
          </div>
        )}

        {/* Toggle Button */}
        <button 
          className={styles.togglePanelsButton}
          onClick={() => setIsPanelsCollapsed(!isPanelsCollapsed)}
          title={isPanelsCollapsed ? "Show panels" : "Hide panels"}
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            {isPanelsCollapsed ? (
              <path d="M9 18l6-6-6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            ) : (
              <path d="M15 18l-6-6 6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            )}
          </svg>
        </button>

        {/* Middle column: ECG Display and Patient Info */}
        <div className={`${styles.middleColumn} ${isPanelsCollapsed ? styles.expanded : ''}`}>
          {/* ECG Display */}
          <div className={styles.ecgDisplayArea}>
            <h2>ECG Signal Display</h2>
            
            {/* View Controls */}
            {!loading && !error && ecgData.length > 0 && (
              <div className={styles.viewControls}>
                {/* View Mode Tabs */}
                <div className={styles.viewModeTabs}>
                  <button
                    className={`${styles.viewModeTab} ${viewMode === 'waveform' ? styles.active : ''}`}
                    onClick={() => setViewMode('waveform')}
                  >
                    Waveform
                  </button>
                  <button
                    className={`${styles.viewModeTab} ${viewMode === 'minimap' ? styles.active : ''}`}
                    onClick={() => setViewMode('minimap')}
                  >
                    Minimap
                  </button>
                  <button
                    className={`${styles.viewModeTab} ${viewMode === 'summary' ? styles.active : ''}`}
                    onClick={() => setViewMode('summary')}
                  >
                    Summary
                  </button>
                </div>

                {/* Zoom Preset Buttons */}
                <div className={styles.zoomControls}>
                  {(Object.keys(ZOOM_PRESETS) as ZoomLevel[]).map((zoomKey) => (
                    <button
                      key={zoomKey}
                      className={`${styles.zoomButton} ${currentZoom === zoomKey ? styles.active : ''}`}
                      onClick={() => setCurrentZoom(zoomKey)}
                    >
                      {ZOOM_PRESETS[zoomKey].label}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {loading && <div className={styles.ecgChartPlaceholder}>Loading ECG Data...</div>}
            {error && <div className={styles.ecgChartPlaceholder} style={{ color: 'red' }}>Error: {error}</div>}
            {!loading && !error && ecgData.length > 0 && (
              <div className={styles.ecgChartWrapper}>
                {predicting && (
                  <div className={styles.predictionBadge}>
                    <div className={styles.badgeSpinner}></div>
                    <span>Analyzing...</span>
                  </div>
                )}
                
                {/* Render based on view mode */}
                {viewMode === 'waveform' && (
                  <EcgChart
                    dataPoints={ecgData}
                    viewWindowSize={viewWindowSize}
                    startIndex={startIndex}
                    predictions={predictions}
                    onSegmentClick={handleSegmentClick}
                    isFullView={currentZoom === 'FULL'}
                  />
                )}
                
                {viewMode === 'minimap' && (
                  <MinimapView
                    dataPoints={ecgData}
                    predictions={predictions}
                    currentPosition={startIndex}
                    viewWindowSize={viewWindowSize}
                    onPositionChange={setStartIndex}
                  />
                )}
                
                {viewMode === 'summary' && (
                  <SummaryChartView
                    predictions={predictions}
                    onMinuteClick={(minute) => {
                      // Jump to that minute in detail view
                      const samplePosition = minute * 6000;
                      setStartIndex(Math.max(0, Math.min(samplePosition, ecgData.length - viewWindowSize)));
                      setViewMode('waveform');
                      setCurrentZoom('DETAIL');
                    }}
                  />
                )}
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
          {!isPanelsCollapsed && (
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
                  <span className={styles.patientValue}>{activeFile}</span>
                </div>
              </div>

              {/* Prediction Summary */}
              <div className={styles.predictionSummary}>
                <h4>Apnea Detection Summary</h4>
                {predicting ? (
                  <div className={styles.predictingStatus}>
                    <div className={styles.smallSpinner}></div>
                    <span>Running prediction...</span>
                  </div>
                ) : predictions.length > 0 ? (
                  <>
                    <div className={styles.summaryStats}>
                      <div className={styles.statItem}>
                        <span className={styles.statLabel}>Minutes Analyzed:</span>
                        <span className={styles.statValue}>{predictions.length}</span>
                      </div>
                      <div className={styles.statItem}>
                        <span className={styles.statLabel}>Apneic Minutes:</span>
                        <span className={`${styles.statValue} ${styles.apneaCount}`}>
                          {predictions.filter(p => p.probability >= 0.5).length}
                        </span>
                      </div>
                      <div className={styles.statItem}>
                        <span className={styles.statLabel}>Status:</span>
                        <span className={`${styles.statValue} ${styles.statusComplete}`}>✓ Complete</span>
                      </div>
                    </div>

                    {/* List of Apneic Minutes with Grad-CAM buttons */}
                    {predictions.filter(p => p.probability >= 0.5).length > 0 && (
                      <div className={styles.apneicMinutesList}>
                        <h5>Detected Apneic Minutes</h5>
                        <div className={styles.minutesList}>
                          {predictions
                            .filter(p => p.probability >= 0.5)
                            .map(pred => (
                              <div key={pred.minute} className={styles.minuteRow}>
                                <span className={styles.minuteLabel}>Minute {pred.minute}</span>
                                <span className={styles.probabilityLabel}>
                                  {(pred.probability * 100).toFixed(0)}%
                                </span>
                                <button 
                                  className={styles.explainButton}
                                  onClick={() => handleSegmentClick(pred.minute)}
                                  disabled={loadingGradcam && selectedMinute === pred.minute}
                                >
                                  {loadingGradcam && selectedMinute === pred.minute ? (
                                    <>
                                      <div className={styles.smallSpinner}></div>
                                      <span>Loading...</span>
                                    </>
                                  ) : gradcamCache.has(pred.minute) ? (
                                    <>👁️ View</>
                                  ) : (
                                    <>🔍 Explain</>
                                  )}
                                </button>
                              </div>
                            ))}
                        </div>
                        
                        {predictions.filter(p => p.probability >= 0.5).length > 1 && (
                          <button 
                            className={styles.generateAllButton}
                            onClick={handleGenerateAllGradcams}
                            disabled={loadingGradcam}
                          >
                            Generate All Explanations
                          </button>
                        )}
                      </div>
                    )}

                    <button 
                      className={styles.rerunButton}
                      onClick={() => fetchPredictions(activeFile)}
                    >
                      Re-run Prediction
                    </button>
                  </>
                ) : (
                  <div className={styles.noPredictions}>
                    <p>No predictions available yet.</p>
                    <button 
                      className={styles.runButton}
                      onClick={() => fetchPredictions(activeFile)}
                    >
                      Run Prediction
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Right column: Chat Window with input */}
        <div className={styles.chatColumn}>
          <div className={styles.chatWindowArea}>
            <h2>AI Assistant</h2>
            <div className={styles.chatMessages}>
              {chatMessages.map((message, index) => (
                <div key={index} className={styles.chatMessage}>
                  <div className={styles.messageHeader}>
                    {message.role.charAt(0).toUpperCase() + message.role.slice(1)}
                  </div>
                  <div className={styles.messageContent}>{message.content}</div>
                </div>
              ))}
              <div ref={chatMessagesEndRef} />
            </div>
          </div>

          {/* Chat Input Box */}
          <div className={styles.chatInputArea}>
            <input
              type="text"
              className={styles.chatInput}
              placeholder="Type your question or command here..."
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyPress={handleKeyPress}
            />
            <button 
              className={styles.sendArrowButton} 
              title="Send"
              onClick={handleSendMessage}
            >
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M22 2L11 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Grad-CAM Visualization Modal */}
      {(gradcamData || (loadingGradcam && selectedMinute !== null)) && (
        <div className={styles.modalOverlay} onClick={() => { setGradcamData(null); setLoadingGradcam(false); }}>
          <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h2>Grad-CAM Explainability - Minute {gradcamData?.minute || selectedMinute}</h2>
              <button 
                className={styles.modalClose}
                onClick={() => { setGradcamData(null); setLoadingGradcam(false); }}
              >
                ×
              </button>
            </div>
            
            <div className={styles.modalBody}>
              {loadingGradcam && !gradcamData ? (
                <div className={styles.loadingIndicator}>
                  <div className={styles.spinner}></div>
                  <p>Generating Grad-CAM explanation for Minute {selectedMinute}...</p>
                </div>
              ) : gradcamData ? (
                <>
                  <div className={styles.gradcamInfo}>
                    <div className={styles.infoItem}>
                      <span className={styles.infoLabel}>Prediction:</span>
                      <span className={`${styles.infoValue} ${gradcamData.predictedClass === 'Apnea' ? styles.apnea : styles.normal}`}>
                        {gradcamData.predictedClass}
                      </span>
                    </div>
                    <div className={styles.infoItem}>
                      <span className={styles.infoLabel}>Confidence:</span>
                      <span className={styles.infoValue}>
                        {(gradcamData.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  
                  <div className={styles.gradcamImageContainer}>
                    <img 
                      src={gradcamData.imageUrl} 
                      alt={`Grad-CAM for minute ${gradcamData.minute}`}
                      className={styles.gradcamImage}
                    />
                  </div>
                  
                  <div className={styles.gradcamExplanation}>
                    <p>
                      <strong>What am I seeing?</strong> This Grad-CAM visualization highlights the regions 
                      of the ECG signal that were most important for the model's prediction. 
                      Warmer colors (red/yellow) indicate areas the model focused on when making its decision.
                    </p>
                  </div>

                  <div className={styles.modalActions}>
                    <button 
                      className={styles.downloadButton}
                      onClick={() => {
                        const link = document.createElement('a');
                        link.href = gradcamData.imageUrl;
                        link.download = `gradcam_minute_${gradcamData.minute}.png`;
                        link.click();
                      }}
                    >
                      Download Image
                    </button>
                  </div>
                </>
              ) : null}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;