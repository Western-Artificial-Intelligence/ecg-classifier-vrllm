import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import styles from '../styles/App.module.css';
import EcgChart, { type GradcamHeatmap } from './EcgChart';
import MinimapView from './MinimapView';
import SummaryChartView from './SummaryChartView';
import { PatientAPI, RecordAPI, AnalysisAPI } from '../services/api';
import type { Patient, Record, PhysiologicalMetrics } from '../types/database';

interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
  timestamp: Date;
}


interface SelectedPatient {
  id: string;
  name: string;
  displayId: string;
}

interface AnalysisLocationState {
  selectedPatient?: unknown;
  patient?: Patient;
  record?: Record;
  filename?: string;
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
type PrimaryToolbarAction = 'run_predictions' | 'run_analysis';

const ZOOM_PRESETS = {
  DETAIL: { label: '1x (Detail)', samples: 500 },
  MINUTE_1: { label: '5x (1 min)', samples: 6000 },
  MINUTE_5: { label: '10x (5 min)', samples: 30000 },
  FULL: { label: 'Full Record', samples: -1 } // -1 means all samples
};

const DEFAULT_SELECTED_PATIENT: SelectedPatient = {
  id: 'default-patient',
  name: 'John Doe',
  displayId: 'P-2024-001'
};

const isSelectedPatient = (value: unknown): value is SelectedPatient => {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const candidate = value as { [key: string]: unknown };
  return (
    typeof candidate.id === 'string' &&
    typeof candidate.name === 'string' &&
    typeof candidate.displayId === 'string'
  );
};

function App() {
  const navigate = useNavigate();
  const location = useLocation();
  const locationState = location.state as AnalysisLocationState | null;
  const selectedPatientCandidate = locationState?.selectedPatient;
  const activePatient = isSelectedPatient(selectedPatientCandidate)
    ? selectedPatientCandidate
    : DEFAULT_SELECTED_PATIENT;
  
  // Database patient and record from navigation
  const dbPatient = locationState?.patient;
  const dbRecord = locationState?.record;
  const initialFilename = locationState?.filename;

  const [ecgData, setEcgData] = useState<number[]>([]);
  const [startIndex, setStartIndex] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [isPatientInfoCollapsed, setIsPatientInfoCollapsed] = useState<boolean>(false);
  const [activeFile, setActiveFile] = useState<string>('a01.dat');
  const [chatInput, setChatInput] = useState<string>('');
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      role: 'system',
      content: 'Welcome to NeuralApnea Triage Assistant. How can I help you analyze this ECG?',
      timestamp: new Date()
    }
  ]);

  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [predicting, setPredicting] = useState<boolean>(false);
  const [analyzingAgent, setAnalyzingAgent] = useState<boolean>(false);

  // Database integration state
  const [currentPatient] = useState<Patient | null>(dbPatient || null);
  const [currentRecord, setCurrentRecord] = useState<Record | null>(dbRecord || null);
  const [, setLoadingPatientRecords] = useState(false);
  const [recordsMap, setRecordsMap] = useState<Map<string, Record>>(new Map());

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
  const [activeGradcamData, setActiveGradcamData] = useState<GradCAMData | null>(null);

  const [currentZoom, setCurrentZoom] = useState<ZoomLevel>('DETAIL');
  const [viewMode, setViewMode] = useState<ViewMode>('waveform');
  const [patientInfoTab, setPatientInfoTab] = useState<'details' | 'predictions' | 'minutes' | 'physio' | 'gradcam'>('details');
  const [gradcamQueue, setGradcamQueue] = useState<Set<number>>(new Set());
  const [gradcamNotifications, setGradcamNotifications] = useState<Array<{ minute: number, id: string }>>([]);
  const [gradcamHeatmapData, setGradcamHeatmapData] = useState<GradcamHeatmap | null>(null);
  const [showAnalysisOverlay] = useState<boolean>(true);
  const [ecgStats, setEcgStats] = useState<ECGStats | null>(null);
  const [generatingAll, setGeneratingAll] = useState<boolean>(false);
  const [generationProgress, setGenerationProgress] = useState<{ current: number, total: number }>({ current: 0, total: 0 });
  const [isChatCollapsed, setIsChatCollapsed] = useState<boolean>(true);
  const [isToolbarExpanded, setIsToolbarExpanded] = useState<boolean>(false);
  
  // Helper function to create cache key
  const getCacheKey = (filename: string, minute: number) => `${filename}_${minute}`;

  // Helper function to convert database metrics to ECGStats format
  const convertMetricsToStats = (dbMetrics: PhysiologicalMetrics): ECGStats => {
    return {
      hrv_time: {
        mean_rri_ms: (dbMetrics.mean_rri_ms ?? 0) as number,
        sdnn_ms: (dbMetrics.sdnn_ms ?? 0) as number,
        rmssd_ms: (dbMetrics.rmssd_ms ?? 0) as number,
        pnn50_percent: (dbMetrics.pnn50_percent ?? 0) as number,
        cv_percent: (dbMetrics.cv_percent ?? 0) as number
      },
      hrv_freq: {
        vlf_power_ms2: (dbMetrics.vlf_power_ms2 ?? 0) as number,
        lf_power_ms2: (dbMetrics.lf_power_ms2 ?? 0) as number,
        hf_power_ms2: (dbMetrics.hf_power_ms2 ?? 0) as number,
        total_power_ms2: (dbMetrics.total_power_ms2 ?? 0) as number,
        lf_hf_ratio: (dbMetrics.lf_hf_ratio ?? 0) as number
      },
      edr: {
        resp_rate_bpm: (dbMetrics.resp_rate_bpm ?? 0) as number,
        edr_amplitude_range: (dbMetrics.edr_amplitude_range ?? 0) as number,
        edr_variability: (dbMetrics.edr_variability ?? 0) as number
      },
      rpeak: {
        num_rpeaks: (dbMetrics.num_rpeaks ?? 0) as number,
        mean_hr_bpm: (dbMetrics.mean_hr_bpm ?? 0) as number,
        hr_std_bpm: (dbMetrics.hr_std_bpm ?? 0) as number,
        recording_duration_min: (dbMetrics.recording_duration_min ?? 0) as number
      }
    };
  };

  const chatMessagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const analysisRequestIdRef = useRef<number>(0);
  const loadingFilesRef = useRef<Set<string>>(new Set());

  // Calculate view window size based on current zoom level
  const viewWindowSize = ZOOM_PRESETS[currentZoom].samples === -1
    ? ecgData.length
    : ZOOM_PRESETS[currentZoom].samples;

  const canUseEcgControls = !loading && !error && ecgData.length > 0;
  const canRunPredictions = !!activeFile && !loading && !error && ecgData.length > 0;
  const primaryToolbarAction: PrimaryToolbarAction = predictions.length > 0 ? 'run_analysis' : 'run_predictions';
  const isToolbarMenuVisible = isToolbarExpanded;

  const handlePatientInfoToggle = () => {
    setIsPatientInfoCollapsed(prev => {
      const nextCollapsed = !prev;
      if (!nextCollapsed) {
        // Opening bottom panel collapses top controls.
        setIsToolbarExpanded(false);
      }
      return nextCollapsed;
    });
  };

  const handleChatToggleFromToolbar = () => {
    setIsChatCollapsed(prev => !prev);
  };

  const handleOpenChat = () => {
    setIsChatCollapsed(false);
  };

  // Scroll to bottom of chat when new messages arrive
  useEffect(() => {
    chatMessagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  // Load patient records from database if patient is provided
  useEffect(() => {
    const loadPatientRecords = async () => {
      if (!currentPatient) return;
      
      console.log('Loading patient records for:', currentPatient.name);
      setLoadingPatientRecords(true);
      try {
        const records = await PatientAPI.getRecords(currentPatient.id);
        console.log('Loaded records:', records);
        
        // Build filename -> Record mapping
        const newMap = new Map<string, Record>();
        records.forEach((record: Record) => {
          newMap.set(`${record.record_name}.dat`, record);
        });
        setRecordsMap(newMap);

        // Auto-load the initial file immediately after loading records
        if (initialFilename) {
          console.log('Auto-loading initial file:', initialFilename);
          handleFileSelect(initialFilename, newMap.get(initialFilename));
        } else if (records.length > 0) {
          const firstFileName = `${records[0].record_name}.dat`;
          handleFileSelect(firstFileName, newMap.get(firstFileName));
        }
      } catch (err) {
        console.error('Failed to load patient records:', err);
        // Keep default files on error
      } finally {
        setLoadingPatientRecords(false);
      }
    };
    
    loadPatientRecords();
  }, [currentPatient]);

  // Fetch Grad-CAM heatmap data when user selects a GradCAM minute (for overlay on ECG chart)
  useEffect(() => {
    if (!activeGradcamData || activeGradcamData.recordingFile !== activeFile) {
      setGradcamHeatmapData(null);
      return;
    }
    let cancelled = false;
    AnalysisAPI.getGradcamHeatmap(activeFile, activeGradcamData.minute)
      .then((data) => {
        if (!cancelled) {
          setGradcamHeatmapData({ minute: data.minute, values: data.heatmap });
        }
      })
      .catch(() => {
        if (!cancelled) setGradcamHeatmapData(null);
      });
    return () => { cancelled = true; };
  }, [activeGradcamData, activeFile]);

  const handleSliderChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setStartIndex(Number(event.target.value));
  };

  const stepAmount = Math.max(1, Math.floor(viewWindowSize / 2));
  const maxStart = Math.max(0, ecgData.length - viewWindowSize);
  const handleTimelineStepLeft = () => {
    setStartIndex((prev) => Math.max(0, prev - stepAmount));
  };
  const handleTimelineStepRight = () => {
    setStartIndex((prev) => Math.min(maxStart, prev + stepAmount));
  };

  const handleFileSelect = async (fileName: string, dbRecordOverride?: Record) => {
    // Prevent duplicate loads
    if (loadingFilesRef.current.has(fileName)) {
      console.log('Already loading', fileName, '- skipping duplicate call');
      return;
    }
    
    loadingFilesRef.current.add(fileName);
    
    try {
      setActiveFile(fileName);

      // Update currentRecord if we have a mapping (use override if provided for immediate access)
      const dbRec = dbRecordOverride || recordsMap.get(fileName);
      if (dbRec) {
        setCurrentRecord(dbRec);
      } else {
        setCurrentRecord(null);
      }

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

            // Try loading from database first using recordsMap
            let cachedPredictions = null;
            let cachedStats = null;

            if (dbRec) {
              console.log('Checking database for predictions - Record ID:', dbRec.id, 'Name:', dbRec.record_name);
              try {
                const dbPreds = await RecordAPI.getPredictions(dbRec.id);
                console.log('Database predictions response:', dbPreds);
                if (dbPreds?.predictions) {
                  console.log('Found predictions in database:', dbPreds.predictions.length);
                  cachedPredictions = dbPreds.predictions;
                }
                
                const dbMetrics = await RecordAPI.getMetrics(dbRec.id);
                console.log('Database metrics response:', dbMetrics);
                if (dbMetrics) {
                  console.log('Found metrics in database');
                  cachedStats = convertMetricsToStats(dbMetrics);
                }
              } catch (e) {
                console.log('Database lookup failed:', e);
              }
            } else {
              console.log('No database record found for file:', fileName);
            }

            // Cache the recording data
            openRecordings.set(fileName, {
              ecgData: result.data || [],
              predictions: cachedPredictions || [],
              stats: cachedStats || null
            });

            // Trigger state updates
            setEcgData(openRecordings.get(fileName)?.ecgData || []);
            setPredictions(cachedPredictions || []);
            setEcgStats(cachedStats);

            // Batch load available Grad-CAM metadata
            await loadGradcamMetadata(fileName);
          } catch (err) {
            setError(`Failed to load ECG data: ${err instanceof Error ? err.message : String(err)}`);
          } finally {
            setLoading(false);
            loadingFilesRef.current.delete(fileName);
          }
        } else {
          // File already cached, just load from cache
          const cached = openRecordings.get(fileName);
          if (cached) {
            setEcgData(cached.ecgData);
            setPredictions(cached.predictions);
            setEcgStats(cached.stats);
          }
          loadingFilesRef.current.delete(fileName);
        }
    } catch (error) {
      console.error('Error in handleFileSelect:', error);
      loadingFilesRef.current.delete(fileName);
    }
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


  // Fetch predictions for the active file
  const fetchPredictions = async (filename: string) => {
    setPredicting(true);
    try {
      const recordName = filename.replace('.dat', '');
      
      // Try loading from database first (if we have a record)
      if (currentRecord && currentRecord.record_name === recordName) {
        try {
          const dbPredictions = await RecordAPI.getPredictions(currentRecord.id);
          if (dbPredictions && dbPredictions.predictions) {
            setPredictions(dbPredictions.predictions);
            
            // Also try to load metrics
            const dbMetrics = await RecordAPI.getMetrics(currentRecord.id);
            if (dbMetrics) {
              // Convert database metrics to ECGStats format (handling optional fields)
              const stats: ECGStats = {
                hrv_time: {
                  mean_rri_ms: (dbMetrics.mean_rri_ms ?? 0) as number,
                  sdnn_ms: (dbMetrics.sdnn_ms ?? 0) as number,
                  rmssd_ms: (dbMetrics.rmssd_ms ?? 0) as number,
                  pnn50_percent: (dbMetrics.pnn50_percent ?? 0) as number,
                  cv_percent: (dbMetrics.cv_percent ?? 0) as number
                },
                hrv_freq: {
                  vlf_power_ms2: (dbMetrics.vlf_power_ms2 ?? 0) as number,
                  lf_power_ms2: (dbMetrics.lf_power_ms2 ?? 0) as number,
                  hf_power_ms2: (dbMetrics.hf_power_ms2 ?? 0) as number,
                  total_power_ms2: (dbMetrics.total_power_ms2 ?? 0) as number,
                  lf_hf_ratio: (dbMetrics.lf_hf_ratio ?? 0) as number
                },
                edr: {
                  resp_rate_bpm: (dbMetrics.resp_rate_bpm ?? 0) as number,
                  edr_amplitude_range: (dbMetrics.edr_amplitude_range ?? 0) as number,
                  edr_variability: (dbMetrics.edr_variability ?? 0) as number
                },
                rpeak: {
                  num_rpeaks: (dbMetrics.num_rpeaks ?? 0) as number,
                  mean_hr_bpm: (dbMetrics.mean_hr_bpm ?? 0) as number,
                  hr_std_bpm: (dbMetrics.hr_std_bpm ?? 0) as number,
                  recording_duration_min: (dbMetrics.recording_duration_min ?? 0) as number
                }
              };
              setEcgStats(stats);
            }
            
            setPredicting(false);
            return;
          }
        } catch (dbErr) {
          console.log('Database lookup failed, will fetch from API:', dbErr);
        }
      }

      // Fetch from API (will generate predictions and auto-save to database if record exists)
      const result = await AnalysisAPI.predict(filename);
      const predictions = result.predictions || [];
      const stats = result.stats || {};

      setPredictions(predictions);
      setEcgStats(stats);

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

  // Handle View button click to show Grad-CAM in right panel
  const handleViewGradCAM = (minute: number) => {
    const cacheKey = getCacheKey(activeFile, minute);
    const imageData = gradcamImages.get(cacheKey);

    if (imageData) {
      setActiveGradcamData({
        minute: imageData.minute,
        imageUrl: imageData.imageUrl,
        probability: imageData.probability,
        predictedClass: imageData.predictedClass,
        recordingFile: activeFile
      });
      setPatientInfoTab('gradcam');
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
        {/* Middle column: ECG Display and Patient Info */}
        <div className={styles.middleColumn}>
          {/* ECG Display */}
          <div className={styles.ecgDisplayArea}>
            {/* Toolbar */}
            <div className={styles.tabBar}>
                <div className={styles.tabBarActions}>
                  <button className={styles.backButton} onClick={() => navigate('/patient-management')}>← Back</button>
                  {!isToolbarMenuVisible && (
                    <button
                      className={styles.toolbarCompactToggle}
                      onClick={() => {
                        // Opening top controls collapses bottom panel.
                        setIsPatientInfoCollapsed(true);
                        setIsToolbarExpanded(true);
                      }}
                      title="Show analysis controls"
                    >
                      Show Controls
                    </button>
                  )}

                  {isToolbarMenuVisible && (
                    <div className={styles.toolbarMenu}>
                    <div className={styles.toolbarControlRow}>
                      <div className={styles.toolbarSelectGroup}>
                        <label htmlFor="viewModeSelect" className={styles.toolbarSelectLabel}>View</label>
                        <select
                          id="viewModeSelect"
                          className={styles.toolbarSelect}
                          value={viewMode}
                          onChange={(e) => {
                            setViewMode(e.target.value as ViewMode);
                          }}
                          disabled={!canUseEcgControls}
                        >
                          <option value="waveform">Waveform</option>
                          <option value="minimap">Minimap</option>
                          <option value="summary">Summary</option>
                        </select>
                      </div>

                      <div className={styles.toolbarSelectGroup}>
                        <label htmlFor="zoomLevelSelect" className={styles.toolbarSelectLabel}>Detail</label>
                        <select
                          id="zoomLevelSelect"
                          className={styles.toolbarSelect}
                          value={currentZoom}
                          onChange={(e) => {
                            setCurrentZoom(e.target.value as ZoomLevel);
                          }}
                          disabled={!canUseEcgControls}
                        >
                          {(Object.keys(ZOOM_PRESETS) as ZoomLevel[]).map((zoomKey) => (
                            <option key={zoomKey} value={zoomKey}>
                              {ZOOM_PRESETS[zoomKey].label}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>

                    <div className={styles.toolbarActionRow}>
                      <button
                        className={styles.runAnalysisButton}
                        onClick={() => fetchPredictions(activeFile)}
                        disabled={!canRunPredictions || predicting}
                      >
                        {predicting ? 'Running Predictions...' : (primaryToolbarAction === 'run_predictions' ? 'Run Predictions' : 'Re-run Predictions')}
                      </button>
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
                      <button
                        className={styles.chatToggleButton}
                        onClick={handleChatToggleFromToolbar}
                        title={isChatCollapsed ? 'Open AI chat panel' : 'Collapse AI chat panel'}
                      >
                        {isChatCollapsed ? 'Open Chat' : 'Hide Chat'}
                      </button>
                      <button
                        className={styles.toolbarCompactToggle}
                        onClick={() => setIsToolbarExpanded(false)}
                        title="Collapse analysis controls"
                      >
                        Collapse Controls
                      </button>
                    </div>
                  </div>
                  )}
                </div>
              </div>

            {/* ECG Content */}
            {loading && <div className={styles.ecgChartPlaceholder}>Loading ECG Data...</div>}
            {error && <div className={styles.ecgChartPlaceholder} style={{ color: 'red' }}>Error: {error}</div>}
            {!loading && !error && ecgData.length === 0 && (
              <div className={styles.ecgChartPlaceholder}>No recording selected. Select a .dat file to display waveform.</div>
            )}
            {!loading && !error && ecgData.length > 0 && (
              <>
                <div className={styles.ecgChartWrapper}>
                  {viewMode === 'waveform' && (
                    <EcgChart
                      dataPoints={ecgData}
                      viewWindowSize={viewWindowSize}
                      startIndex={startIndex}
                      predictions={predictions}
                      onSegmentClick={handleSegmentClick}
                      isFullView={currentZoom === 'FULL'}
                      showAnnotations={showAnalysisOverlay}
                      gradcamHeatmap={gradcamHeatmapData}
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
                        const samplePosition = minute * 6000;
                        setStartIndex(Math.max(0, Math.min(samplePosition, ecgData.length - viewWindowSize)));
                        setViewMode('waveform');
                        setCurrentZoom('DETAIL');
                      }}
                    />
                  )}
                </div>

                <div className={styles.ecgTimelineControl}>
                  <div className={styles.timelineSliderRow}>
                    <button
                      type="button"
                      className={styles.timelineArrowButton}
                      onClick={handleTimelineStepLeft}
                      disabled={!canUseEcgControls || startIndex <= 0}
                      title={`Move left by half view (${stepAmount} samples)`}
                      aria-label="Move chart left"
                    >
                      ←
                    </button>
                    <input
                      type="range"
                      min={0}
                      max={maxStart}
                      value={startIndex}
                      onChange={handleSliderChange}
                      className={styles.timelineSlider}
                      disabled={ecgData.length <= viewWindowSize}
                    />
                    <button
                      type="button"
                      className={styles.timelineArrowButton}
                      onClick={handleTimelineStepRight}
                      disabled={!canUseEcgControls || startIndex >= maxStart}
                      title={`Move right by half view (${stepAmount} samples)`}
                      aria-label="Move chart right"
                    >
                      →
                    </button>
                  </div>
                  <p>Viewing samples {startIndex} to {Math.min(startIndex + viewWindowSize, ecgData.length)} of {ecgData.length}</p>
                </div>
              </>
            )}
          </div>

          <div className={styles.patientInfoToggleRow}>
            <button
              className={styles.patientInfoToggleButton}
              onClick={handlePatientInfoToggle}
              title={isPatientInfoCollapsed ? 'Show patient information panel' : 'Hide patient information panel'}
            >
              {isPatientInfoCollapsed ? 'Show Patient Info' : 'Hide Patient Info'}
            </button>
          </div>

          {/* Patient Information Box */}
          {!isPatientInfoCollapsed && (
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
                <button
                  className={`${styles.tabButton} ${patientInfoTab === 'gradcam' ? styles.active : ''}`}
                  onClick={() => setPatientInfoTab('gradcam')}
                >
                  GradCAM
                </button>
              </div>

              {/* Tab Content */}
              <div className={styles.tabContent}>
                {/* Details Tab */}
                {patientInfoTab === 'details' && (
                  <div className={styles.patientDetails}>
                    <div className={styles.patientInfoRow}>
                      <span className={styles.patientLabel}>Patient Name:</span>
                      <span className={styles.patientValue}>{currentPatient?.name || activePatient.name}</span>
                    </div>
                    <div className={styles.patientInfoRow}>
                      <span className={styles.patientLabel}>Patient #:</span>
                      <span className={styles.patientValue}>
                        {currentPatient ? `ID: ${currentPatient.id}` : activePatient.displayId}
                      </span>
                    </div>
                    <div className={styles.patientInfoRow}>
                      <span className={styles.patientLabel}>File:</span>
                      <span className={styles.patientValue}>{activeFile}</span>
                    </div>
                    {currentPatient && (
                      <>
                        {currentPatient.date_of_birth && (
                          <div className={styles.patientInfoRow}>
                            <span className={styles.patientLabel}>DOB:</span>
                            <span className={styles.patientValue}>{currentPatient.date_of_birth}</span>
                          </div>
                        )}
                        {currentPatient.gender && (
                          <div className={styles.patientInfoRow}>
                            <span className={styles.patientLabel}>Gender:</span>
                            <span className={styles.patientValue}>{currentPatient.gender}</span>
                          </div>
                        )}
                        {currentPatient.weight_kg && (
                          <div className={styles.patientInfoRow}>
                            <span className={styles.patientLabel}>Weight:</span>
                            <span className={styles.patientValue}>{currentPatient.weight_kg} kg</span>
                          </div>
                        )}
                        {currentPatient.height_cm && (
                          <div className={styles.patientInfoRow}>
                            <span className={styles.patientLabel}>Height:</span>
                            <span className={styles.patientValue}>{currentPatient.height_cm} cm</span>
                          </div>
                        )}
                      </>
                    )}
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
                        <p>No predictions available yet. Use the toolbar action to run predictions.</p>
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
                      </div>
                    )}
                  </div>
                )}
                {/* GradCAM Tab */}
                {patientInfoTab === 'gradcam' && (
                  <div className={styles.gradcamPanel}>
                    {activeGradcamData ? (
                      <>
                        <div className={styles.gradcamPanelHeader}>
                          <h5>Grad-CAM — Minute {activeGradcamData.minute}</h5>
                          <div className={styles.gradcamPanelMeta}>
                            <span className={`${styles.statValue} ${activeGradcamData.predictedClass === 'Apnea' ? styles.apneaCount : styles.statusComplete}`}>
                              {activeGradcamData.predictedClass}
                            </span>
                            <span className={styles.statValue}>
                              {(activeGradcamData.probability * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                        <img
                          src={activeGradcamData.imageUrl}
                          alt={`Grad-CAM for minute ${activeGradcamData.minute}`}
                          className={styles.gradcamPanelImage}
                          loading="lazy"
                        />
                        <button
                          className={styles.downloadButton}
                          onClick={() => {
                            const link = document.createElement('a');
                            link.href = activeGradcamData.imageUrl;
                            link.download = `gradcam_minute_${activeGradcamData.minute}.png`;
                            link.click();
                          }}
                        >
                          Download Image
                        </button>
                      </>
                    ) : (
                      <div className={styles.noMetrics}>
                        <p>No GradCAM selected. Click "View" on an apneic minute to display its explanation here.</p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Right column: Chat Window with input */}
        <div className={`${styles.chatColumn} ${isChatCollapsed ? styles.collapsed : ''}`}>
          {isChatCollapsed ? (
            <button
              className={styles.chatCollapsedToggle}
              onClick={handleOpenChat}
              title="Open AI assistant panel"
            >
              AI
            </button>
          ) : (
            <>
              <div className={styles.chatWindowArea}>
                <div className={styles.chatWindowHeader}>
                  <h2>AI Assistant</h2>
                  <button
                    className={styles.chatToggleButton}
                    onClick={() => setIsChatCollapsed(true)}
                    title="Collapse AI chat panel"
                  >
                    Hide
                  </button>
                </div>
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
            </>
          )}
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

