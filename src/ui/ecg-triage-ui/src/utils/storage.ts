// IndexedDB wrapper for analysis results
const DB_NAME = 'ECGAnalysisDB';
const DB_VERSION = 1;
const PREDICTIONS_STORE = 'predictions';
const GRADCAM_STORE = 'gradcams';

interface Prediction {
  minute: number;
  probability: number;
}

interface StoredPrediction {
  filename: string;
  timestamp: number;
  predictions: Prediction[];
}

interface GradCAMData {
  minute: number;
  imageUrl: string;
  probability: number;
  predictedClass: string;
}

interface StoredGradCAM {
  filename: string;
  minute: number;
  imageData: string; // base64
  probability: number;
  predictedClass: string;
  timestamp: number;
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

      // Create Grad-CAM store
      if (!db.objectStoreNames.contains(GRADCAM_STORE)) {
        const gradcamStore = db.createObjectStore(GRADCAM_STORE, { keyPath: ['filename', 'minute'] });
        gradcamStore.createIndex('filename', 'filename', { unique: false });
        gradcamStore.createIndex('timestamp', 'timestamp', { unique: false });
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

// Save Grad-CAM
export const saveGradCAM = async (filename: string, minute: number, data: GradCAMData): Promise<void> => {
  const db = await initDB();
  
  return new Promise((resolve, reject) => {
    const transaction = db.transaction([GRADCAM_STORE], 'readwrite');
    const store = transaction.objectStore(GRADCAM_STORE);
    
    const storedData: StoredGradCAM = {
      filename,
      minute,
      imageData: data.imageUrl,
      probability: data.probability,
      predictedClass: data.predictedClass,
      timestamp: Date.now()
    };
    
    const request = store.put(storedData);
    
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
    
    transaction.oncomplete = () => db.close();
  });
};

// Load Grad-CAM
export const loadGradCAM = async (filename: string, minute: number): Promise<GradCAMData | null> => {
  const db = await initDB();
  
  return new Promise((resolve, reject) => {
    const transaction = db.transaction([GRADCAM_STORE], 'readonly');
    const store = transaction.objectStore(GRADCAM_STORE);
    
    const request = store.get([filename, minute]);
    
    request.onsuccess = () => {
      const result = request.result as StoredGradCAM | undefined;
      if (result) {
        resolve({
          minute: result.minute,
          imageUrl: result.imageData,
          probability: result.probability,
          predictedClass: result.predictedClass
        });
      } else {
        resolve(null);
      }
    };
    request.onerror = () => reject(request.error);
    
    transaction.oncomplete = () => db.close();
  });
};

// Load all Grad-CAMs for a file
export const loadAllGradCAMs = async (filename: string): Promise<Map<number, GradCAMData>> => {
  const db = await initDB();
  
  return new Promise((resolve, reject) => {
    const transaction = db.transaction([GRADCAM_STORE], 'readonly');
    const store = transaction.objectStore(GRADCAM_STORE);
    const index = store.index('filename');
    
    const request = index.getAll(filename);
    
    request.onsuccess = () => {
      const results = request.result as StoredGradCAM[];
      const gradcamMap = new Map<number, GradCAMData>();
      
      results.forEach(item => {
        gradcamMap.set(item.minute, {
          minute: item.minute,
          imageUrl: item.imageData,
          probability: item.probability,
          predictedClass: item.predictedClass
        });
      });
      
      resolve(gradcamMap);
    };
    request.onerror = () => reject(request.error);
    
    transaction.oncomplete = () => db.close();
  });
};
