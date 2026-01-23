import React, { useState, useEffect, useRef } from 'react';
import styles from './styles/App.module.css';
import EcgChart from './components/EcgChart';

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

  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatMessagesEndRef = useRef<HTMLDivElement>(null);
  const viewWindowSize = 500;

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
    </div>
  );
}

export default App;