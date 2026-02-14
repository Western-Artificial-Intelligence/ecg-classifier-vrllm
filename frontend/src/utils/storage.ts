// IndexedDB wrapper for analysis results
const DB_NAME = 'ECGAnalysisDB';
const DB_VERSION = 3;  // Incremented to remove gradcams store
const PREDICTIONS_STORE = 'predictions';
const STATS_STORE = 'stats';

interface Prediction {
  minute: number;
  probability: number;
}

interface StoredPrediction {
  filename: string;
  timestamp: number;
  predictions: Prediction[];
}


interface StoredStats {
  filename: string;
  timestamp: number;
  stats: any; // ECGStats type
}

// Initialize DB
export const initDB = (): Promise<IDBDatabase> => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;

      // Create predictions store
      if (!db.objectStoreNames.contains(PREDICTIONS_STORE)) {
        const predictionsStore = db.createObjectStore(PREDICTIONS_STORE, { keyPath: 'filename' });
        predictionsStore.createIndex('timestamp', 'timestamp', { unique: false });
      }

      // Remove Grad-CAM store if it exists (no longer needed)
      if (db.objectStoreNames.contains('gradcams')) {
        db.deleteObjectStore('gradcams');
      }

      // Create Stats store
      if (!db.objectStoreNames.contains(STATS_STORE)) {
        const statsStore = db.createObjectStore(STATS_STORE, { keyPath: 'filename' });
        statsStore.createIndex('timestamp', 'timestamp', { unique: false });
      }
    };
  });
};

// Save predictions
export const savePredictions = async (filename: string, predictions: Prediction[]): Promise<void> => {
  const db = await initDB();
  
  return new Promise((resolve, reject) => {
    const transaction = db.transaction([PREDICTIONS_STORE], 'readwrite');
    const store = transaction.objectStore(PREDICTIONS_STORE);
    
    const data: StoredPrediction = {
      filename,
      timestamp: Date.now(),
      predictions
    };
    
    const request = store.put(data);
    
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
    
    transaction.oncomplete = () => db.close();
  });
};

// Load predictions
export const loadPredictions = async (filename: string): Promise<Prediction[] | null> => {
  const db = await initDB();
  
  return new Promise((resolve, reject) => {
    const transaction = db.transaction([PREDICTIONS_STORE], 'readonly');
    const store = transaction.objectStore(PREDICTIONS_STORE);
    
    const request = store.get(filename);
    
    request.onsuccess = () => {
      const result = request.result as StoredPrediction | undefined;
      resolve(result ? result.predictions : null);
    };
    request.onerror = () => reject(request.error);
    
    transaction.oncomplete = () => db.close();
  });
};

// Save stats
export const saveStats = async (filename: string, stats: any): Promise<void> => {
  const db = await initDB();
  
  return new Promise((resolve, reject) => {
    const transaction = db.transaction([STATS_STORE], 'readwrite');
    const store = transaction.objectStore(STATS_STORE);
    
    const data: StoredStats = {
      filename,
      timestamp: Date.now(),
      stats
    };
    
    const request = store.put(data);
    
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
    
    transaction.oncomplete = () => db.close();
  });
};

// Load stats
export const loadStats = async (filename: string): Promise<any | null> => {
  const db = await initDB();
  
  return new Promise((resolve, reject) => {
    const transaction = db.transaction([STATS_STORE], 'readonly');
    const store = transaction.objectStore(STATS_STORE);
    
    const request = store.get(filename);
    
    request.onsuccess = () => {
      const result = request.result as StoredStats | undefined;
      resolve(result ? result.stats : null);
    };
    request.onerror = () => reject(request.error);
    
    transaction.oncomplete = () => db.close();
  });
};
