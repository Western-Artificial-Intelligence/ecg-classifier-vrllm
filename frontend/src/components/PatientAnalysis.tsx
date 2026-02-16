import React, { useState, useEffect, useRef } from 'react';
import styles from '../styles/App.module.css';
import EcgChart from './EcgChart';
import MinimapView from './MinimapView';
import SummaryChartView from './SummaryChartView';
import { savePredictions, loadPredictions, saveStats, loadStats } from '../utils/storage';

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
  recordingFile?: string;
}

interface Tab {
  id: string;
  type: 'ecg' | 'gradcam';
  title: string;

  // For ECG tabs
  ecgFile?: string;
  viewMode?: ViewMode;
  startIndex?: number;
  zoom?: ZoomLevel;

  // For Grad-CAM tabs
  minute?: number;
  gradcamData?: GradCAMData;
}

interface ECGStats {
  hrv_time?: {
    mean_rri_ms: number;
    sdnn_ms: number;
    rmssd_ms: number;
    pnn50_percent: number;
    cv_percent: number;
  };
  hrv_freq?: {
    vlf_power_ms2: number;
    lf_power_ms2: number;
    hf_power_ms2: number;
    total_power_ms2: number;
    lf_hf_ratio: number;
  };
  edr?: {
    resp_rate_bpm: number;
    edr_amplitude_range: number;
    edr_variability: number;
  };
  rpeak?: {
    num_rpeaks: number;
    mean_hr_bpm: number;
    hr_std_bpm: number;
    recording_duration_min: number;
  };
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
      content: 'Welcome to NeuralApnea Triage Assistant. How can I help you analyze this ECG?',
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
  const [analyzingAgent, setAnalyzingAgent] = useState<boolean>(false);

  // New cache structure for tab system
  interface RecordingCache {
    ecgData: number[];
    predictions: Prediction[];
    stats: ECGStats | null;
  }

  interface GradCAMCache {
    minute: number;
    imageUrl: string;
    probability: number;
    predictedClass: string;
    recordingFile: string;
  }

  const [openRecordings, setOpenRecordings] = useState<Map<string, RecordingCache>>(new Map());
  const [gradcamImages, setGradcamImages] = useState<Map<string, GradCAMCache>>(new Map());
  const [openGradcamTabs, setOpenGradcamTabs] = useState<Set<string>>(new Set());

  // Tab system state
  const [tabs, setTabs] = useState<Tab[]>([]);
  const [activeTabId, setActiveTabId] = useState<string | null>(null);
  const [hoveredTabId, setHoveredTabId] = useState<string | null>(null);

  const [currentZoom, setCurrentZoom] = useState<ZoomLevel>('DETAIL');
  const [viewMode, setViewMode] = useState<ViewMode>('waveform');
  const [patientInfoTab, setPatientInfoTab] = useState<'details' | 'predictions' | 'minutes' | 'physio'>('details');
  const [gradcamQueue, setGradcamQueue] = useState<Set<number>>(new Set());
  const [gradcamNotifications, setGradcamNotifications] = useState<Array<{ minute: number, id: string }>>([]);
  const [showAnalysisOverlay, setShowAnalysisOverlay] = useState<boolean>(true);
  const [ecgStats, setEcgStats] = useState<ECGStats | null>(null);
  const [generatingAll, setGeneratingAll] = useState<boolean>(false);
  const [generationProgress, setGenerationProgress] = useState<{ current: number, total: number }>({ current: 0, total: 0 });

  // Helper function to create cache key
  const getCacheKey = (filename: string, minute: number) => `${filename}_${minute}`;

  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatMessagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const analysisRequestIdRef = useRef<number>(0);

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

  // Get the currently active tab
  const activeTab = tabs.find(t => t.id === activeTabId);

  // Scroll to bottom of chat when new messages arrive
  useEffect(() => {
    chatMessagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  // Synchronize active tab data with display state
  useEffect(() => {
    if (activeTab && activeTab.type === 'ecg' && activeTab.ecgFile) {
      // Update activeFile to match the tab
      setActiveFile(activeTab.ecgFile);

      // Load data from cache if available
      const cached = openRecordings.get(activeTab.ecgFile);
      if (cached) {
        setEcgData(cached.ecgData);
        setPredictions(cached.predictions);
        setEcgStats(cached.stats);
      }

      // Restore tab-specific view settings
      if (activeTab.viewMode) setViewMode(activeTab.viewMode);
      if (activeTab.zoom) setCurrentZoom(activeTab.zoom);
      if (activeTab.startIndex !== undefined) setStartIndex(activeTab.startIndex);
    }
  }, [activeTabId, activeTab, openRecordings]);


  const handleSliderChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newIndex = Number(event.target.value);
    setStartIndex(newIndex);
    // Update the active tab's position
    updateActiveTab({ startIndex: newIndex });
  };

  const handleFileSelect = async (fileName: string) => {
    setActiveFile(fileName);

    // Check if tab already exists for this file
    const existingTab = tabs.find(t => t.type === 'ecg' && t.ecgFile === fileName);

    if (existingTab) {
      // Tab exists, just switch to it
      setActiveTabId(existingTab.id);
    } else {
      // New tab - create it
      createECGTab(fileName);

      // Load and cache recording data if not already cached
      if (!openRecordings.has(fileName)) {
        try {
          // Load ECG data
          setLoading(true);
          const response = await fetch(`http://localhost:8000/api/ecg_data/${fileName}`);
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          const result = await response.json();

          // Load predictions and stats from IndexedDB if available
          let cachedPredictions = await loadPredictions(fileName);
          let cachedStats = await loadStats(fileName);

          // If no predictions but we have ECG data, try fetching from API
          if (!cachedPredictions || cachedPredictions.length === 0) {
            // Check if predictions exist on backend
            try {
              const recordName = fileName.replace('.dat', '');
              const predResponse = await fetch(`http://localhost:8000/api/predict/${recordName}`, {
                method: 'POST'
              });

              if (predResponse.ok) {
                const predResult = await predResponse.json();
                const fetchedPredictions = predResult.predictions || [];
                const fetchedStats = predResult.stats || null;

                cachedPredictions = fetchedPredictions;
                cachedStats = fetchedStats;

                // Save to IndexedDB for future
                if (fetchedPredictions.length > 0) {
                  await savePredictions(fileName, fetchedPredictions);
                }
                if (fetchedStats) {
                  await saveStats(fileName, fetchedStats);
                }
              }
            } catch (e) {
              console.error('Failed to fetch predictions:', e);
            }
          }

          // Cache the recording data
          setOpenRecordings(prev => new Map(prev).set(fileName, {
            ecgData: result.data || [],
            predictions: cachedPredictions || [],
            stats: cachedStats || null
          }));

          // Update state for rendering
          setEcgData(result.data || []);
          setPredictions(cachedPredictions || []);
          setEcgStats(cachedStats || null);

          // Batch load available Grad-CAM metadata
          await loadGradcamMetadata(fileName);

          setLoading(false);
        } catch (e: any) {
          console.error('Failed to load recording:', e);
          setError(e.message);
          setLoading(false);
        }
      } else {
        // Recording already cached, load from cache
        const cached = openRecordings.get(fileName)!;
        setEcgData(cached.ecgData);
        setPredictions(cached.predictions);
        setEcgStats(cached.stats);
        setLoading(false);
      }
    }
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

  const appendAssistantMessage = (content: string) => {
    const assistantMessage: ChatMessage = {
      role: 'assistant',
      content,
      timestamp: new Date()
    };
    setChatMessages(prev => [...prev, assistantMessage]);
  };

  const renderInlineMarkdown = (text: string): React.ReactNode[] => {
    const parts: React.ReactNode[] = [];
    const pattern = /(\*\*[^*]+\*\*|`[^`]+`)/g;
    let lastIndex = 0;
    let match: RegExpExecArray | null;
    let key = 0;

    while ((match = pattern.exec(text)) !== null) {
      if (match.index > lastIndex) {
        parts.push(text.slice(lastIndex, match.index));
      }

      const token = match[0];
      if (token.startsWith('**') && token.endsWith('**')) {
        parts.push(<strong key={`b-${key++}`}>{token.slice(2, -2)}</strong>);
      } else if (token.startsWith('`') && token.endsWith('`')) {
        parts.push(<code key={`c-${key++}`} className={styles.inlineCode}>{token.slice(1, -1)}</code>);
      } else {
        parts.push(token);
      }

      lastIndex = pattern.lastIndex;
    }

    if (lastIndex < text.length) {
      parts.push(text.slice(lastIndex));
    }

    return parts;
  };

  const renderMarkdownMessage = (content: string): React.ReactNode => {
    const lines = content.split('\n');
    const nodes: React.ReactNode[] = [];
    let inCodeBlock = false;
    let codeBuffer: string[] = [];
    let listBuffer: string[] = [];
    let key = 0;

    const flushList = () => {
      if (listBuffer.length === 0) return;
      nodes.push(
        <ul key={`ul-${key++}`} className={styles.markdownList}>
          {listBuffer.map((item, idx) => (
            <li key={`li-${idx}`}>{renderInlineMarkdown(item)}</li>
          ))}
        </ul>
      );
      listBuffer = [];
    };

    const flushCode = () => {
      if (codeBuffer.length === 0) return;
      nodes.push(
        <pre key={`pre-${key++}`} className={styles.codeBlock}>
          <code>{codeBuffer.join('\n')}</code>
        </pre>
      );
      codeBuffer = [];
    };

    for (const line of lines) {
      if (line.trim().startsWith('```')) {
        if (inCodeBlock) {
          flushCode();
          inCodeBlock = false;
        } else {
          flushList();
          inCodeBlock = true;
        }
        continue;
      }

      if (inCodeBlock) {
        codeBuffer.push(line);
        continue;
      }

      if (line.trim().startsWith('- ')) {
        listBuffer.push(line.trim().slice(2));
        continue;
      }

      flushList();

      const trimmed = line.trim();
      if (!trimmed) {
        nodes.push(<div key={`sp-${key++}`} className={styles.markdownSpacer}></div>);
        continue;
      }

      if (trimmed.startsWith('### ')) {
        nodes.push(
          <h5 key={`h3-${key++}`} className={styles.markdownH3}>
            {renderInlineMarkdown(trimmed.slice(4))}
          </h5>
        );
        continue;
      }

      if (trimmed.startsWith('## ')) {
        nodes.push(
          <h4 key={`h2-${key++}`} className={styles.markdownH2}>
            {renderInlineMarkdown(trimmed.slice(3))}
          </h4>
        );
        continue;
      }

      if (trimmed.startsWith('# ')) {
        nodes.push(
          <h3 key={`h1-${key++}`} className={styles.markdownH1}>
            {renderInlineMarkdown(trimmed.slice(2))}
          </h3>
        );
        continue;
      }

      nodes.push(
        <p key={`p-${key++}`} className={styles.markdownParagraph}>
          {renderInlineMarkdown(trimmed)}
        </p>
      );
    }

    flushList();
    flushCode();

    return nodes;
  };

  const handleKeyPress = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  // Load all available Grad-CAM images for a recording
  const loadGradcamMetadata = async (filename: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/gradcam/list/${filename}`);
      const result = await response.json();

      // Update gradcamImages cache for this recording
      setGradcamImages(prev => {
        const newCache = new Map(prev);

        result.images.forEach((img: any) => {
          const key = getCacheKey(filename, img.minute);
          newCache.set(key, {
            minute: img.minute,
            imageUrl: img.image_url,
            probability: img.probability,
            predictedClass: img.predicted_class,
            recordingFile: filename
          });
        });

        return newCache;
      });
    } catch (e) {
      console.error('Failed to load Grad-CAM metadata:', e);
    }
  };

  // Tab Management Functions
  const createECGTab = (filename: string) => {
    // Check if tab already exists for this file
    const existing = tabs.find(t => t.type === 'ecg' && t.ecgFile === filename);
    if (existing) {
      setActiveTabId(existing.id);
      return;
    }

    const tabId = `ecg-${filename}-${Date.now()}`;
    const newTab: Tab = {
      id: tabId,
      type: 'ecg',
      title: filename,
      ecgFile: filename,
      viewMode: 'waveform',
      startIndex: 0,
      zoom: 'DETAIL'
    };
    setTabs(prev => [...prev, newTab]);
    setActiveTabId(tabId);
  };

  const createGradCAMTab = (minute: number, gradcamData: GradCAMData) => {
    // Check if tab already exists for this minute
    const existing = tabs.find(t =>
      t.type === 'gradcam' &&
      t.minute === minute &&
      gradcamData.recordingFile === activeFile
    );
    if (existing) {
      setActiveTabId(existing.id);
      return;
    }

    // Extract recording name without .dat extension
    const recordingName = activeFile.replace('.dat', '');

    const tabId = `gradcam-${minute}-${Date.now()}`;
    const newTab: Tab = {
      id: tabId,
      type: 'gradcam',
      title: `${recordingName} Min ${minute}`,
      minute,
      gradcamData
    };
    setTabs(prev => [...prev, newTab]);
    setActiveTabId(tabId);
  };

  const closeTab = (tabId: string) => {
    const tab = tabs.find(t => t.id === tabId);
    if (!tab) return;

    if (tab.type === 'ecg') {
      // Closing a recording tab
      const filename = tab.ecgFile!;

      // Remove recording from cache
      setOpenRecordings(prev => {
        const newCache = new Map(prev);
        newCache.delete(filename);
        return newCache;
      });

      // Remove Grad-CAM images ONLY if no Grad-CAM tabs are open for them
      setGradcamImages(prev => {
        const newCache = new Map(prev);

        // Get all images for this recording
        Array.from(newCache.entries()).forEach(([key, data]) => {
          if (data.recordingFile === filename) {
            // Only delete if no open tab for this image
            if (!openGradcamTabs.has(key)) {
              newCache.delete(key);
            }
          }
        });

        return newCache;
      });

    } else if (tab.type === 'gradcam') {
      // Closing a Grad-CAM tab
      const cacheKey = getCacheKey(activeFile, tab.minute!);

      // Remove from openGradcamTabs
      setOpenGradcamTabs(prev => {
        const newSet = new Set(prev);
        newSet.delete(cacheKey);
        return newSet;
      });

      // Check if parent recording is still open
      const parentRecordingOpen = tabs.some(t =>
        t.type === 'ecg' &&
        t.ecgFile === gradcamImages.get(cacheKey)?.recordingFile
      );

      // If parent recording is closed, remove the image from cache
      if (!parentRecordingOpen) {
        setGradcamImages(prev => {
          const newCache = new Map(prev);
          newCache.delete(cacheKey);
          return newCache;
        });
      }
    }

    // Remove tab from tabs array
    setTabs(prev => prev.filter(t => t.id !== tabId));

    // Switch to another tab if closing active tab
    if (activeTabId === tabId) {
      const remainingTabs = tabs.filter(t => t.id !== tabId);
      setActiveTabId(remainingTabs.length > 0 ? remainingTabs[0].id : null);
    }
  };

  const updateActiveTab = (updates: Partial<Tab>) => {
    setTabs(prev => prev.map(t =>
      t.id === activeTabId ? { ...t, ...updates } : t
    ));
  };

  // Fetch predictions for the active file
  const fetchPredictions = async (filename: string) => {
    setPredicting(true);
    try {
      // Try loading from IndexedDB first
      const cached = await loadPredictions(filename);
      if (cached) {
        setPredictions(cached);
        // Also load cached stats if available
        const cachedStats = await loadStats(filename);
        if (cachedStats) {
          setEcgStats(cachedStats);
        }
        setPredicting(false);
        return;
      }

      // Fetch from API
      const recordName = filename.replace('.dat', '');
      const response = await fetch(`http://localhost:8000/api/predict/${recordName}`, {
        method: 'POST'
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      const predictions = result.predictions || [];
      const stats = result.stats || {};

      setPredictions(predictions);
      setEcgStats(stats);

      // Save to IndexedDB
      await savePredictions(filename, predictions);
      await saveStats(filename, stats);

      // Update the cache for this recording
      setOpenRecordings(prev => {
        const newCache = new Map(prev);
        const existing = newCache.get(filename);
        if (existing) {
          newCache.set(filename, {
            ...existing,
            predictions,
            stats
          });
        }
        return newCache;
      });
    } catch (e: any) {
      console.error('Error fetching predictions:', e);
      setError(`Prediction error: ${e.message}`);
    } finally {
      setPredicting(false);
    }
  };

  const runAgentAnalysis = async (filename: string) => {
    if (predicting || analyzingAgent) return;

    if (!predictions || predictions.length === 0) {
      appendAssistantMessage('Run predictions first, then run analysis.');
      return;
    }

    const requestId = analysisRequestIdRef.current + 1;
    analysisRequestIdRef.current = requestId;

    setAnalyzingAgent(true);
    try {
      const recordName = filename.replace('.dat', '');
      const response = await fetch(`http://localhost:8000/api/agent/analyze/${recordName}`, {
        method: 'POST'
      });

      let result: any = null;
      try {
        result = await response.json();
      } catch {
        result = null;
      }

      const fallbackMessage = `Analysis is currently unavailable for "${recordName}". Predictions are still available.`;
      const analysisMessage = typeof result?.analysis === 'string' && result.analysis.trim().length > 0
        ? result.analysis
        : fallbackMessage;

      if (analysisRequestIdRef.current === requestId) {
        appendAssistantMessage(analysisMessage);
      }
    } catch (e) {
      console.error('Error running agent analysis:', e);
      if (analysisRequestIdRef.current === requestId) {
        appendAssistantMessage(
          `Analysis is currently unavailable for "${filename.replace('.dat', '')}". Predictions are still available.`
        );
      }
    } finally {
      if (analysisRequestIdRef.current === requestId) {
        setAnalyzingAgent(false);
      }
    }
  };

  // Handle segment click to generate Grad-CAM silently (non-blocking)
  const handleSegmentClick = async (minute: number) => {
    const cacheKey = getCacheKey(activeFile, minute);

    // Check if already cached
    if (gradcamImages.has(cacheKey)) {
      // Already cached, nothing to do
      return;
    }

    // Add to processing queue (non-blocking)
    setGradcamQueue(prev => new Set(prev).add(minute));

    // Process in background
    try {
      const recordName = activeFile.replace('.dat', '');
      const response = await fetch(
        `http://localhost:8000/api/gradcam/${recordName}?minute=${minute}`,
        { method: 'POST' }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      // IMMEDIATELY cache the image (backend already saved it to disk)
      setGradcamImages(prev => {
        const newCache = new Map(prev);
        newCache.set(cacheKey, {
          minute: result.minute,
          imageUrl: result.image_url,
          probability: result.probability,
          predictedClass: result.predicted_class,
          recordingFile: activeFile
        });
        return newCache;
      });

      // Show notification
      const notifId = `gradcam_${minute}_${Date.now()}`;
      setGradcamNotifications(prev => [...prev, { minute, id: notifId }]);

      // Auto-remove notification after 5 seconds
      setTimeout(() => {
        setGradcamNotifications(prev => prev.filter(n => n.id !== notifId));
      }, 5000);

    } catch (e: any) {
      console.error(`Failed to generate Grad-CAM for minute ${minute}:`, e);
    } finally {
      // Remove from queue
      setGradcamQueue(prev => {
        const newSet = new Set(prev);
        newSet.delete(minute);
        return newSet;
      });
    }
  };

  // Handle View button click to create Grad-CAM tab
  const handleViewGradCAM = (minute: number) => {
    const cacheKey = getCacheKey(activeFile, minute);
    const imageData = gradcamImages.get(cacheKey);

    if (imageData) {
      // Create tab and mark as open
      createGradCAMTab(minute, {
        minute: imageData.minute,
        imageUrl: imageData.imageUrl,
        probability: imageData.probability,
        predictedClass: imageData.predictedClass
      });

      // Track that this Grad-CAM tab is now open
      setOpenGradcamTabs(prev => new Set(prev).add(cacheKey));
    } else {
      alert('Please generate the Grad-CAM explanation first by clicking "Explain".');
    }
  };

  // Generate all Grad-CAMs for apneic minutes with cancellation support
  const handleGenerateAllGradcams = async () => {
    const apneicMinutes = predictions
      .filter(p => p.probability >= 0.5)
      .map(p => p.minute)
      .filter(minute => !gradcamImages.has(getCacheKey(activeFile, minute))); // Only process uncached minutes

    if (apneicMinutes.length === 0) {
      return;
    }

    setGeneratingAll(true);
    setGenerationProgress({ current: 0, total: apneicMinutes.length });

    // Create abort controller
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      let completed = 0;

      // Process sequentially to allow cancellation
      for (const minute of apneicMinutes) {
        if (controller.signal.aborted) {
          break;
        }

        const cacheKey = getCacheKey(activeFile, minute);

        // Check if already cached (might have been generated while processing)
        if (gradcamImages.has(cacheKey)) {
          completed++;
          setGenerationProgress({ current: completed, total: apneicMinutes.length });
          continue;
        }

        // Add to processing queue
        setGradcamQueue(prev => new Set(prev).add(minute));

        try {
          const recordName = activeFile.replace('.dat', '');
          const response = await fetch(
            `http://localhost:8000/api/gradcam/${recordName}?minute=${minute}`,
            {
              method: 'POST',
              signal: controller.signal
            }
          );

          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }

          const result = await response.json();

          // Cache the image
          setGradcamImages(prev => {
            const newCache = new Map(prev);
            newCache.set(cacheKey, {
              minute: result.minute,
              imageUrl: result.image_url,
              probability: result.probability,
              predictedClass: result.predicted_class,
              recordingFile: activeFile
            });
            return newCache;
          });

          completed++;
          setGenerationProgress({ current: completed, total: apneicMinutes.length });

        } catch (e: any) {
          if (e.name === 'AbortError') {
            console.log('Generation cancelled');
            break;
          }
          console.error(`Failed to generate Grad-CAM for minute ${minute}:`, e);
        } finally {
          // Remove from queue
          setGradcamQueue(prev => {
            const newSet = new Set(prev);
            newSet.delete(minute);
            return newSet;
          });
        }
      }
    } finally {
      setGeneratingAll(false);
      setGenerationProgress({ current: 0, total: 0 });
      abortControllerRef.current = null;
    }
  };

  const handleCancelGeneration = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };




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
              <path d="M9 18l6-6-6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            ) : (
              <path d="M15 18l-6-6 6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            )}
          </svg>
        </button>

        {/* Middle column: ECG Display and Patient Info */}
        <div className={`${styles.middleColumn} ${isPanelsCollapsed ? styles.expanded : ''}`}>
          {/* ECG Display */}
          <div className={styles.ecgDisplayArea}>
            {/* Tab Bar */}
            {tabs.length > 0 && (
              <div className={styles.tabBar}>
                <div className={styles.tabList}>
                  {tabs.map(tab => (
                    <div
                      key={tab.id}
                      className={`${styles.tab} ${activeTabId === tab.id ? styles.active : ''}`}
                      onClick={() => setActiveTabId(tab.id)}
                      onMouseEnter={() => setHoveredTabId(tab.id)}
                      onMouseLeave={() => setHoveredTabId(null)}
                    >
                      <span className={styles.tabIcon}>
                        {tab.type === 'ecg' ? <span className={styles.ecgIcon}>R</span> : '🔍'}
                      </span>
                      <span className={styles.tabTitle}>{tab.title}</span>
                      {hoveredTabId === tab.id && (
                        <button
                          className={styles.tabClose}
                          onClick={(e) => {
                            e.stopPropagation();
                            closeTab(tab.id);
                          }}
                        >
                          ×
                        </button>
                      )}
                    </div>
                  ))}
                </div>
                <div className={styles.tabBarActions}>
                  {predicting ? (
                    <div className={styles.predictionBadge}>
                      <div className={styles.badgeSpinner}></div>
                      <span>Running predictions...</span>
                    </div>
                  ) : predictions.length === 0 && activeFile && !loading && !error && ecgData.length > 0 ? (
                    <button
                      className={styles.runAnalysisButton}
                      onClick={() => fetchPredictions(activeFile)}
                    >
                      Run Predictions
                    </button>
                  ) : null}
                </div>
              </div>
            )}

            {/* View Controls - Only show for ECG tabs */}
            {activeTab && activeTab.type === 'ecg' && !loading && !error && (
              <div className={styles.viewControls}>
                {/* View Mode Tabs */}
                <div className={styles.viewModeTabs}>
                  <button
                    className={`${styles.viewModeTab} ${(activeTab?.viewMode || viewMode) === 'waveform' ? styles.active : ''}`}
                    onClick={() => {
                      setViewMode('waveform');
                      updateActiveTab({ viewMode: 'waveform' });
                    }}
                  >
                    Waveform
                  </button>
                  <button
                    className={`${styles.viewModeTab} ${(activeTab?.viewMode || viewMode) === 'minimap' ? styles.active : ''}`}
                    onClick={() => {
                      setViewMode('minimap');
                      updateActiveTab({ viewMode: 'minimap' });
                    }}
                  >
                    Minimap
                  </button>
                  <button
                    className={`${styles.viewModeTab} ${(activeTab?.viewMode || viewMode) === 'summary' ? styles.active : ''}`}
                    onClick={() => {
                      setViewMode('summary');
                      updateActiveTab({ viewMode: 'summary' });
                    }}
                  >
                    Summary
                  </button>
                </div>

                {/* Zoom Preset Buttons */}
                <div className={styles.zoomControls}>
                  {(Object.keys(ZOOM_PRESETS) as ZoomLevel[]).map((zoomKey) => (
                    <button
                      key={zoomKey}
                      className={`${styles.zoomButton} ${(activeTab?.zoom || currentZoom) === zoomKey ? styles.active : ''}`}
                      onClick={() => {
                        setCurrentZoom(zoomKey);
                        updateActiveTab({ zoom: zoomKey });
                      }}
                    >
                      {ZOOM_PRESETS[zoomKey].label}
                    </button>
                  ))}
                </div>

                <button
                  className={styles.agentActionButton}
                  onClick={() => runAgentAnalysis(activeFile)}
                  disabled={predicting || analyzingAgent || predictions.length === 0}
                  title={
                    predictions.length === 0
                      ? 'Run predictions before analysis'
                      : 'Run AI agent analysis'
                  }
                >
                  {analyzingAgent ? 'Running Analysis...' : 'Run Analysis'}
                </button>

                {/* Analysis Overlay Toggle */}
                {ecgData.length > 0 && (
                  <button
                    className={`${styles.overlayToggle} ${showAnalysisOverlay ? styles.active : ''}`}
                    onClick={() => setShowAnalysisOverlay(!showAnalysisOverlay)}
                    title={showAnalysisOverlay ? "Hide analysis overlay" : "Show analysis overlay"}
                  >
                    {showAnalysisOverlay ? '👁️ Hide Overlay' : '👁️ Show Overlay'}
                  </button>
                )}

              </div>
            )}

            {/* Content rendering based on active tab */}
            {activeTab && activeTab.type === 'ecg' && (
              <>
                {loading && <div className={styles.ecgChartPlaceholder}>Loading ECG Data...</div>}
                {error && <div className={styles.ecgChartPlaceholder} style={{ color: 'red' }}>Error: {error}</div>}
                {!loading && !error && ecgData.length > 0 && (
                  <div className={styles.ecgChartWrapper}>
                    {/* Render based on view mode */}
                    {(activeTab?.viewMode || viewMode) === 'waveform' && (
                      <EcgChart
                        dataPoints={ecgData}
                        viewWindowSize={viewWindowSize}
                        startIndex={startIndex}
                        predictions={predictions}
                        onSegmentClick={handleSegmentClick}
                        isFullView={currentZoom === 'FULL'}
                        showAnnotations={showAnalysisOverlay}
                      />
                    )}

                    {(activeTab?.viewMode || viewMode) === 'minimap' && (
                      <MinimapView
                        dataPoints={ecgData}
                        predictions={predictions}
                        currentPosition={startIndex}
                        viewWindowSize={viewWindowSize}
                        onPositionChange={setStartIndex}
                      />
                    )}

                    {(activeTab?.viewMode || viewMode) === 'summary' && (
                      <SummaryChartView
                        predictions={predictions}
                        onMinuteClick={(minute) => {
                          // Jump to that minute in detail view
                          const samplePosition = minute * 6000;
                          setStartIndex(Math.max(0, Math.min(samplePosition, ecgData.length - viewWindowSize)));
                          setViewMode('waveform');
                          setCurrentZoom('DETAIL');
                          updateActiveTab({ viewMode: 'waveform', zoom: 'DETAIL', startIndex: samplePosition });
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
              </>
            )}

            {/* Grad-CAM Tab Rendering */}
            {activeTab && activeTab.type === 'gradcam' && activeTab.gradcamData && (
              <div className={styles.gradcamViewer}>
                <div className={styles.gradcamHeader}>
                  <h3>Grad-CAM Explainability - Minute {activeTab.minute}</h3>
                  <div className={styles.gradcamMeta}>
                    <span className={`${styles.prediction} ${activeTab.gradcamData.predictedClass === 'Apnea' ? styles.apnea : styles.normal}`}>
                      {activeTab.gradcamData.predictedClass}
                    </span>
                    <span className={styles.confidence}>
                      {(activeTab.gradcamData.probability * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                <div className={styles.gradcamImageWrapper}>
                  <img
                    src={activeTab.gradcamData.imageUrl}
                    alt={`Grad-CAM for minute ${activeTab.minute}`}
                    className={styles.gradcamFullImage}
                    loading="lazy"
                  />
                </div>

                <div className={styles.gradcamActions}>
                  <button
                    className={styles.downloadButton}
                    onClick={() => {
                      const link = document.createElement('a');
                      link.href = activeTab.gradcamData!.imageUrl;
                      link.download = `gradcam_minute_${activeTab.minute}.png`;
                      link.click();
                    }}
                  >
                    Download Image
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Patient Information Box */}
          {!isPanelsCollapsed && (
            <div className={styles.patientInfoBox}>
              <h3>Patient Information</h3>

              {/* Tab Headers */}
              <div className={styles.tabHeaders}>
                <button
                  className={`${styles.tabButton} ${patientInfoTab === 'details' ? styles.active : ''}`}
                  onClick={() => setPatientInfoTab('details')}
                >
                  Details
                </button>
                <button
                  className={`${styles.tabButton} ${patientInfoTab === 'predictions' ? styles.active : ''}`}
                  onClick={() => setPatientInfoTab('predictions')}
                >
                  Predictions
                </button>
                <button
                  className={`${styles.tabButton} ${patientInfoTab === 'minutes' ? styles.active : ''}`}
                  onClick={() => setPatientInfoTab('minutes')}
                >
                  Apneic Minutes
                </button>
                <button
                  className={`${styles.tabButton} ${patientInfoTab === 'physio' ? styles.active : ''}`}
                  onClick={() => setPatientInfoTab('physio')}
                >
                  Physiological Metrics
                </button>
              </div>

              {/* Tab Content */}
              <div className={styles.tabContent}>
                {/* Details Tab */}
                {patientInfoTab === 'details' && (
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
                )}

                {/* Predictions Tab */}
                {patientInfoTab === 'predictions' && (
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
                )}

                {/* Apneic Minutes Tab */}
                {patientInfoTab === 'minutes' && (
                  <div className={styles.apneicMinutesTab}>
                    {predicting ? (
                      <div className={styles.predictingStatus}>
                        <div className={styles.smallSpinner}></div>
                        <span>Running prediction...</span>
                      </div>
                    ) : predictions.filter(p => p.probability >= 0.5).length > 0 ? (
                      <>
                        <h5>Detected Apneic Minutes ({predictions.filter(p => p.probability >= 0.5).length})</h5>
                        <div className={styles.minutesList}>
                          {predictions
                            .filter(p => p.probability >= 0.5)
                            .map(pred => (
                              <div key={pred.minute} className={styles.minuteRow}>
                                <span className={styles.minuteLabel}>Minute {pred.minute}</span>
                                <span className={styles.probabilityLabel}>
                                  {(pred.probability * 100).toFixed(0)}%
                                </span>
                                {gradcamQueue.has(pred.minute) ? (
                                  <button
                                    className={styles.explainButton}
                                    disabled={true}
                                  >
                                    <div className={styles.smallSpinner}></div>
                                    <span>Processing...</span>
                                  </button>
                                ) : gradcamImages.has(getCacheKey(activeFile, pred.minute)) ? (
                                  <button
                                    className={styles.viewButton}
                                    onClick={() => handleViewGradCAM(pred.minute)}
                                  >
                                    👁️ View
                                  </button>
                                ) : (
                                  <button
                                    className={styles.explainButton}
                                    onClick={() => handleSegmentClick(pred.minute)}
                                  >
                                    🔍 Explain
                                  </button>
                                )}
                              </div>
                            ))}
                        </div>

                        {predictions.filter(p => p.probability >= 0.5).length > 1 && (
                          generatingAll ? (
                            <button
                              className={styles.cancelButton}
                              onClick={handleCancelGeneration}
                            >
                              ⏹️ Cancel Generation ({generationProgress.current}/{generationProgress.total})
                            </button>
                          ) : (
                            <button
                              className={styles.generateAllButton}
                              onClick={handleGenerateAllGradcams}
                            >
                              Generate All Explanations
                            </button>
                          )
                        )}
                      </>
                    ) : (
                      <div className={styles.noApneicMinutes}>
                        <p>No apneic minutes detected in this record.</p>
                      </div>
                    )}
                  </div>
                )}

                {/* Physiological Metrics Tab */}
                {patientInfoTab === 'physio' && (
                  <div className={styles.physioMetrics}>
                    {ecgStats && Object.keys(ecgStats).length > 0 ? (
                      <>
                        {/* R-peak Stats */}
                        {ecgStats.rpeak && (
                          <div className={styles.metricSection}>
                            <h5>R-peak Detection</h5>
                            <div className={styles.metricRow}>
                              <span>Total R-peaks:</span>
                              <span>{ecgStats.rpeak.num_rpeaks}</span>
                            </div>
                            <div className={styles.metricRow}>
                              <span>Mean HR:</span>
                              <span>{ecgStats.rpeak.mean_hr_bpm.toFixed(1)} bpm</span>
                            </div>
                            <div className={styles.metricRow}>
                              <span>HR Std Dev:</span>
                              <span>{ecgStats.rpeak.hr_std_bpm.toFixed(1)} bpm</span>
                            </div>
                            <div className={styles.metricRow}>
                              <span>Duration:</span>
                              <span>{ecgStats.rpeak.recording_duration_min.toFixed(1)} min</span>
                            </div>
                          </div>
                        )}

                        {/* HRV Time Domain */}
                        {ecgStats.hrv_time && (
                          <div className={styles.metricSection}>
                            <h5>HRV - Time Domain</h5>
                            <div className={styles.metricRow}>
                              <span>Mean RRI:</span>
                              <span>{ecgStats.hrv_time.mean_rri_ms.toFixed(1)} ms</span>
                            </div>
                            <div className={styles.metricRow}>
                              <span>SDNN:</span>
                              <span>{ecgStats.hrv_time.sdnn_ms.toFixed(1)} ms</span>
                            </div>
                            <div className={styles.metricRow}>
                              <span>RMSSD:</span>
                              <span>{ecgStats.hrv_time.rmssd_ms.toFixed(1)} ms</span>
                            </div>
                            <div className={styles.metricRow}>
                              <span>pNN50:</span>
                              <span>{ecgStats.hrv_time.pnn50_percent.toFixed(1)}%</span>
                            </div>
                          </div>
                        )}

                        {/* HRV Frequency Domain */}
                        {ecgStats.hrv_freq && (
                          <div className={styles.metricSection}>
                            <h5>HRV - Frequency Domain</h5>
                            <div className={styles.metricRow}>
                              <span>VLF Power:</span>
                              <span>{ecgStats.hrv_freq.vlf_power_ms2.toFixed(1)} ms²</span>
                            </div>
                            <div className={styles.metricRow}>
                              <span>LF Power:</span>
                              <span>{ecgStats.hrv_freq.lf_power_ms2.toFixed(1)} ms²</span>
                            </div>
                            <div className={styles.metricRow}>
                              <span>HF Power:</span>
                              <span>{ecgStats.hrv_freq.hf_power_ms2.toFixed(1)} ms²</span>
                            </div>
                            <div className={styles.metricRow}>
                              <span>LF/HF Ratio:</span>
                              <span>{ecgStats.hrv_freq.lf_hf_ratio.toFixed(2)}</span>
                            </div>
                            <div className={styles.metricRow}>
                              <span>Total Power:</span>
                              <span>{ecgStats.hrv_freq.total_power_ms2.toFixed(1)} ms²</span>
                            </div>
                          </div>
                        )}

                        {/* EDR */}
                        {ecgStats.edr && (
                          <div className={styles.metricSection}>
                            <h5>ECG-Derived Respiration</h5>
                            <div className={styles.metricRow}>
                              <span>Respiratory Rate:</span>
                              <span>{ecgStats.edr.resp_rate_bpm.toFixed(1)} breaths/min</span>
                            </div>
                            <div className={styles.metricRow}>
                              <span>EDR Range:</span>
                              <span>{ecgStats.edr.edr_amplitude_range.toFixed(3)}</span>
                            </div>
                            <div className={styles.metricRow}>
                              <span>EDR Variability:</span>
                              <span>{ecgStats.edr.edr_variability.toFixed(3)}</span>
                            </div>
                          </div>
                        )}
                      </>
                    ) : (
                      <div className={styles.noMetrics}>
                        <p>No physiological metrics available yet.</p>
                        {!predicting && predictions.length === 0 && (
                          <button
                            className={styles.runButton}
                            onClick={() => fetchPredictions(activeFile)}
                          >
                            Run Analysis
                          </button>
                        )}
                      </div>
                    )}
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
                  <div className={styles.messageContent}>{renderMarkdownMessage(message.content)}</div>
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
                <path d="M22 2L11 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Toast Notifications */}
      <div className={styles.notificationContainer}>
        {gradcamNotifications.map(notif => (
          <div key={notif.id} className={styles.toast}>
            <span>✓ Grad-CAM for Minute {notif.minute} ready</span>
            <button
              className={styles.toastButton}
              onClick={() => handleViewGradCAM(notif.minute)}
            >
              View
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
