import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

export type Patient = {
  id: string;
  displayName: string;
  patientCode: string;
  age?: number;
  sex?: "F" | "M" | "X";
  bmi?: number;
  createdAt: number;
};

export type EcgUpload = {
  id: string;
  patientId: string;
  name: string;
  uploadedAt: number;
  mimeType: string;
  size: number;
  fs?: number;
  note?: string;
};

function uid(prefix = "id") {
  return `${prefix}-${Math.random().toString(16).slice(2)}-${Date.now().toString(16)}`;
}

export function formatDateTime(ts: number) {
  try {
    return new Date(ts).toLocaleString(undefined, {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit"
    });
  } catch {
    return String(ts);
  }
}

export function formatBytes(n: number) {
  if (!Number.isFinite(n)) return "-";
  const units = ["B", "KB", "MB", "GB"];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

// ---- IndexedDB (renderer-side) ----
type EcgDbRecord = EcgUpload & { blob: Blob };
const DB_NAME = "somniscope";
const DB_VERSION = 1;

function req<T>(r: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}

function txDone(t: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    t.oncomplete = () => resolve();
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  });
}

async function openDb(): Promise<IDBDatabase> {
  return await new Promise((resolve, reject) => {
    const r = indexedDB.open(DB_NAME, DB_VERSION);
    r.onupgradeneeded = () => {
      const db = r.result;
      if (!db.objectStoreNames.contains("patients")) db.createObjectStore("patients", { keyPath: "id" });
      if (!db.objectStoreNames.contains("ecgs")) {
        const s = db.createObjectStore("ecgs", { keyPath: "id" });
        s.createIndex("byPatient", "patientId", { unique: false });
      }
    };
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}

async function dbGetAllPatients(): Promise<Patient[]> {
  const db = await openDb();
  const t = db.transaction("patients", "readonly");
  const out = (await req(t.objectStore("patients").getAll())) as Patient[];
  db.close();
  return out;
}

async function dbGetAllEcgs(): Promise<EcgUpload[]> {
  const db = await openDb();
  const t = db.transaction("ecgs", "readonly");
  const rows = (await req(t.objectStore("ecgs").getAll())) as EcgDbRecord[];
  db.close();
  return rows.map(({ blob: _blob, ...meta }) => meta);
}

async function dbUpsertPatient(p: Patient): Promise<void> {
  const db = await openDb();
  const t = db.transaction("patients", "readwrite");
  t.objectStore("patients").put(p);
  await txDone(t);
  db.close();
}

async function dbDeletePatient(patientId: string): Promise<void> {
  const db = await openDb();
  const t = db.transaction(["patients", "ecgs"], "readwrite");
  t.objectStore("patients").delete(patientId);

  const ecgsStore = t.objectStore("ecgs");
  const idx = ecgsStore.index("byPatient");
  const ecgs = (await req(idx.getAll(patientId))) as EcgDbRecord[];
  for (const e of ecgs) ecgsStore.delete(e.id);

  await txDone(t);
  db.close();
}

async function dbUpsertEcg(meta: EcgUpload, blob: Blob): Promise<void> {
  const db = await openDb();
  const t = db.transaction("ecgs", "readwrite");
  t.objectStore("ecgs").put({ ...meta, blob } satisfies EcgDbRecord);
  await txDone(t);
  db.close();
}

async function dbDeleteEcg(ecgId: string): Promise<void> {
  const db = await openDb();
  const t = db.transaction("ecgs", "readwrite");
  t.objectStore("ecgs").delete(ecgId);
  await txDone(t);
  db.close();
}

export async function dbGetEcgBlob(ecgId: string): Promise<Blob | null> {
  const db = await openDb();
  const t = db.transaction("ecgs", "readonly");
  const rec = (await req(t.objectStore("ecgs").get(ecgId))) as EcgDbRecord | undefined;
  db.close();
  return rec?.blob ?? null;
}

// ---- React state / actions ----
type PatientsState = {
  isLoaded: boolean;
  patients: Patient[];
  ecgs: EcgUpload[];
  selectedPatientId: string | null;
};

type PatientsActions = {
  selectPatient: (patientId: string | null) => void;
  createPatient: (p: Omit<Patient, "id" | "createdAt">) => Promise<Patient>;
  updatePatient: (patient: Patient) => Promise<void>;
  removePatient: (patientId: string) => Promise<void>;
  addEcgUpload: (patientId: string, file: File, meta?: { fs?: number; note?: string }) => Promise<EcgUpload>;
  removeEcgUpload: (ecgId: string) => Promise<void>;
};

type PatientsContextValue = PatientsState & PatientsActions;
const PatientsContext = createContext<PatientsContextValue | null>(null);

const LS_SELECTED_PATIENT = "somni.selectedPatientId";

function readSelectedPatientId(): string | null {
  try {
    const v = localStorage.getItem(LS_SELECTED_PATIENT);
    return v && v.length ? v : null;
  } catch {
    return null;
  }
}

function writeSelectedPatientId(v: string | null) {
  try {
    if (!v) localStorage.removeItem(LS_SELECTED_PATIENT);
    else localStorage.setItem(LS_SELECTED_PATIENT, v);
  } catch {
    // ignore
  }
}

export function PatientsProvider({ children }: { children: React.ReactNode }) {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [ecgs, setEcgs] = useState<EcgUpload[]>([]);
  const [isLoaded, setIsLoaded] = useState(false);
  const [selectedPatientId, setSelectedPatientId] = useState<string | null>(() => readSelectedPatientId());

  useEffect(() => writeSelectedPatientId(selectedPatientId), [selectedPatientId]);

  useEffect(() => {
    let alive = true;
    (async () => {
      const [ps, es] = await Promise.all([dbGetAllPatients(), dbGetAllEcgs()]);
      if (!alive) return;

      let finalPatients = ps;
      if (ps.length === 0) {
        const demo: Patient[] = [
          {
            id: uid("pt"),
            displayName: "A. Nguyen",
            patientCode: "SS-01942",
            age: 58,
            sex: "F",
            bmi: 31.2,
            createdAt: Date.now()
          },
          {
            id: uid("pt"),
            displayName: "J. Patel",
            patientCode: "SS-02108",
            age: 46,
            sex: "M",
            bmi: 27.5,
            createdAt: Date.now()
          }
        ];
        await Promise.all(demo.map((p) => dbUpsertPatient(p)));
        if (!alive) return;
        finalPatients = demo;
        setPatients(demo);
      } else {
        setPatients(ps);
      }

      setEcgs(es);

      const selected = readSelectedPatientId();
      const exists = selected ? finalPatients.some((p) => p.id === selected) : true;
      if (!exists && selected) setSelectedPatientId(null);

      setIsLoaded(true);
    })().catch(() => setIsLoaded(true));

    return () => {
      alive = false;
    };
  }, []);

  const selectPatient = useCallback((patientId: string | null) => setSelectedPatientId(patientId), []);

  const createPatient = useCallback(async (p: Omit<Patient, "id" | "createdAt">) => {
    const next: Patient = { ...p, id: uid("pt"), createdAt: Date.now() };
    await dbUpsertPatient(next);
    setPatients((prev) => [next, ...prev]);
    return next;
  }, []);

  const updatePatient = useCallback(async (patient: Patient) => {
    await dbUpsertPatient(patient);
    setPatients((prev) => prev.map((p) => (p.id === patient.id ? patient : p)));
  }, []);

  const removePatient = useCallback(async (patientId: string) => {
    await dbDeletePatient(patientId);
    setPatients((prev) => prev.filter((p) => p.id !== patientId));
    setEcgs((prev) => prev.filter((e) => e.patientId !== patientId));
    setSelectedPatientId((cur) => (cur === patientId ? null : cur));
  }, []);

  const addEcgUpload = useCallback(
    async (patientId: string, file: File, meta?: { fs?: number; note?: string }) => {
      const rec: EcgUpload = {
        id: uid("ecg"),
        patientId,
        name: file.name || "ECG Upload",
        uploadedAt: Date.now(),
        mimeType: file.type || "application/octet-stream",
        size: file.size,
        fs: meta?.fs,
        note: meta?.note
      };
      await dbUpsertEcg(rec, file);
      setEcgs((prev) => [rec, ...prev]);
      return rec;
    },
    []
  );

  const removeEcgUpload = useCallback(async (ecgId: string) => {
    await dbDeleteEcg(ecgId);
    setEcgs((prev) => prev.filter((e) => e.id !== ecgId));
  }, []);

  const value = useMemo<PatientsContextValue>(
    () => ({
      isLoaded,
      patients,
      ecgs,
      selectedPatientId,
      selectPatient,
      createPatient,
      updatePatient,
      removePatient,
      addEcgUpload,
      removeEcgUpload
    }),
    [isLoaded, patients, ecgs, selectedPatientId, selectPatient, createPatient, updatePatient, removePatient, addEcgUpload, removeEcgUpload]
  );

  return <PatientsContext.Provider value={value}>{children}</PatientsContext.Provider>;
}

export function usePatients() {
  const ctx = useContext(PatientsContext);
  if (!ctx) throw new Error("usePatients must be used within PatientsProvider");
  return ctx;
}


