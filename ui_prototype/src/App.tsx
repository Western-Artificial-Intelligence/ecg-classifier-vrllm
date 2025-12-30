import React, { useMemo, useState } from "react";
import { Waveform, type Region, synthEcg, parseEcgBlob } from "./ecg";
import { PatientsProvider, dbGetEcgBlob, formatBytes, formatDateTime, usePatients } from "./patients";

type Route =
  | { name: "dashboard" }
  | { name: "patients" }
  | { name: "patient"; patientId: string };

function Sidebar({
  active,
  onNavigate
}: {
  active: "dashboard" | "patients";
  onNavigate: (dest: "dashboard" | "patients") => void;
}) {
  return (
    <aside className="sidebar" aria-label="Primary">
      <div className="brand">
        <div className="brandTitle">SomniScope</div>
        <div className="brandSub">Sleep screening UI</div>
      </div>

      <div className="nav">
        <button className="navBtn" aria-current={active === "dashboard" ? "page" : undefined} onClick={() => onNavigate("dashboard")}>
          <div>Dashboard</div>
          <div className="navHint">Signal overview + explainability</div>
        </button>
        <button className="navBtn" aria-current={active === "patients" ? "page" : undefined} onClick={() => onNavigate("patients")}>
          <div>Patients</div>
          <div className="navHint">Directory + uploads</div>
        </button>
      </div>

      <div className="status">
        <div className="label">Status</div>
        <div className="statusRow">
          <span className="dot" aria-hidden />
          <span>Ready</span>
        </div>
        <div className="subtle">Local-only demo storage (IndexedDB).</div>
      </div>
    </aside>
  );
}

function Topbar({
  title,
  subtitle,
  right
}: {
  title: string;
  subtitle?: string;
  right?: React.ReactNode;
}) {
  return (
    <div className="topbar">
      <div style={{ minWidth: 0 }}>
        <h1 className="h1">{title}</h1>
        {subtitle ? <div className="subtle">{subtitle}</div> : null}
      </div>
      {right ? <div className="row">{right}</div> : null}
    </div>
  );
}

function DashboardView() {
  const { patients, ecgs, selectedPatientId } = usePatients();
  const activePatient = selectedPatientId ? patients.find((p) => p.id === selectedPatientId) ?? null : null;
  const activeEcgCount = activePatient ? ecgs.filter((e) => e.patientId === activePatient.id).length : 0;

  const secondsTotal = 60;
  const fs = 250;
  const pxPerSecond = 24;
  const samples = useMemo(() => synthEcg({ seconds: secondsTotal, fs, seed: 42 }), []);

  const regions = useMemo<Region[]>(
    () => [
      {
        id: "ap-1",
        t0: 9.8,
        t1: 15.6,
        confidence: 0.81,
        reason: "Respiratory modulation + transient bradycardic response; reduced beat-to-beat variability."
      },
      {
        id: "ap-2",
        t0: 27.2,
        t1: 33.4,
        confidence: 0.74,
        reason: "Amplitude instability with recovery overshoot; pattern aligns with obstructive event candidate."
      },
      {
        id: "ap-3",
        t0: 44.9,
        t1: 52.3,
        confidence: 0.88,
        reason: "Sustained irregularity with low SQI micro-artifacts; confidence boosted by contextual HRV shift."
      }
    ],
    []
  );

  const [selectedRegion, setSelectedRegion] = useState<Region | null>(null);
  const [msgs, setMsgs] = useState<{ id: string; role: "assistant" | "user"; text: string }[]>(() => [
    {
      id: "m0",
      role: "assistant",
      text: "Click a highlighted ECG segment to generate a focused explanation (placeholder)."
    }
  ]);

  const activePatientLabel = activePatient ? `${activePatient.displayName} • ${activePatient.patientCode}` : "No active patient";

  return (
    <div className="grid2">
      <section className="panel panelPad">
        <Topbar
          title="Dashboard"
          subtitle={`${activePatientLabel} • ECGs: ${activePatient ? String(activeEcgCount) : "—"} • ${formatDateTime(Date.now())}`}
          right={
            <button className="btn btnGhost" onClick={() => setSelectedRegion(null)}>
              Clear selection
            </button>
          }
        />

        <div className="label">ECG overview</div>
        <div className="waveWrap" style={{ marginTop: 10 }}>
          <div className="waveScroll">
            <Waveform
              samples={samples}
              fs={fs}
              pxPerSecond={pxPerSecond}
              height={420}
              regions={regions}
              onClickRegion={(r) => {
                setSelectedRegion(r);
                const pct = Math.round(r.confidence * 100);
                setMsgs((prev) => [
                  ...prev,
                  {
                    id: `auto-${r.id}-${Date.now()}`,
                    role: "assistant",
                    text: `Segment ${r.t0.toFixed(1)}s–${r.t1.toFixed(1)}s • Confidence ${pct}%\n${r.reason}`
                  }
                ]);
              }}
            />
          </div>
        </div>

        <div className="row" style={{ marginTop: 10, justifyContent: "space-between" }}>
          <div className="mono">0.0s</div>
          <div className="mono">Lead I • preview</div>
          <div className="mono">{secondsTotal.toFixed(1)}s</div>
        </div>
      </section>

      <section className="panel panelPad" aria-label="Chat">
        <Topbar title="Assistant" subtitle={selectedRegion ? `Selected: ${selectedRegion.t0.toFixed(1)}s–${selectedRegion.t1.toFixed(1)}s` : "No segment selected"} />

        <div className="row" style={{ marginBottom: 10 }}>
          {["Explain segment", "Generate report", "Next steps"].map((c) => (
            <button
              key={c}
              className="btn"
              onClick={() => setMsgs((p) => [...p, { id: `${c}-${Date.now()}`, role: "user", text: c }])}
            >
              {c}
            </button>
          ))}
        </div>

        <div className="panel" style={{ background: "var(--panel2)", padding: 10, height: 420, overflow: "auto" }} role="log" aria-live="polite">
          {msgs.map((m) => (
            <div key={m.id} style={{ marginBottom: 8 }}>
              <div
                style={{
                  maxWidth: "92%",
                  padding: "8px 10px",
                  borderRadius: 10,
                  border: "1px solid var(--border)",
                  background: m.role === "assistant" ? "rgba(0,0,0,0.35)" : "rgba(255,255,255,0.06)",
                  marginLeft: m.role === "assistant" ? 0 : "auto",
                  whiteSpace: "pre-wrap"
                }}
              >
                {m.text}
              </div>
            </div>
          ))}
        </div>

        <div className="row" style={{ marginTop: 10 }}>
          <input
            className="input"
            style={{ flex: 1 }}
            placeholder="Ask about a segment…"
            onKeyDown={(e) => {
              if (e.key !== "Enter") return;
              const input = e.currentTarget;
              const v = input.value.trim();
              if (!v) return;
              input.value = "";
              setMsgs((p) => [...p, { id: `u-${Date.now()}`, role: "user", text: v }]);
            }}
          />
          <button
            className="btn"
            onClick={() =>
              setMsgs((p) => [
                ...p,
                { id: `a-${Date.now()}`, role: "assistant", text: "Placeholder response (wire this to your backend later)." }
              ])
            }
          >
            Send
          </button>
        </div>
      </section>
    </div>
  );
}

function PatientsView({ onOpenPatient }: { onOpenPatient: (patientId: string) => void }) {
  const { patients, ecgs, createPatient, removePatient, selectPatient, isLoaded } = usePatients();
  const [q, setQ] = useState("");
  const [draftName, setDraftName] = useState("");
  const [draftCode, setDraftCode] = useState("");

  const rows = useMemo(() => {
    const term = q.trim().toLowerCase();
    return patients
      .filter((p) => {
        if (!term) return true;
        const hay = `${p.displayName} ${p.patientCode}`.toLowerCase();
        return hay.includes(term);
      })
      .map((p) => {
        const count = ecgs.filter((e) => e.patientId === p.id).length;
        const lastEcgAt = ecgs
          .filter((e) => e.patientId === p.id)
          .reduce<number | null>((max, e) => (max == null ? e.uploadedAt : Math.max(max, e.uploadedAt)), null);
        return { p, count, lastEcgAt };
      })
      .sort((a, b) => (b.lastEcgAt ?? -1) - (a.lastEcgAt ?? -1) || (b.p.createdAt ?? 0) - (a.p.createdAt ?? 0));
  }, [patients, ecgs, q]);

  return (
    <section className="panel panelPad">
      <Topbar
        title="Patients"
        subtitle="Directory + uploads (local IndexedDB)"
        right={
          <>
            <input className="input" style={{ width: 280 }} placeholder="Search name or ID…" value={q} onChange={(e) => setQ(e.target.value)} />
          </>
        }
      />

      <div className="row" style={{ marginBottom: 10 }}>
        <div className="mono">{isLoaded ? `${rows.length} shown` : "Loading…"}</div>
      </div>

      <div className="panel" style={{ background: "var(--panel2)" }}>
        <div className="scroll" style={{ maxHeight: "56vh" }}>
          {!isLoaded ? (
            <div style={{ padding: 12, color: "var(--muted)" }}>Loading…</div>
          ) : rows.length === 0 ? (
            <div style={{ padding: 12, color: "var(--muted)" }}>No patients match your search.</div>
          ) : (
            <table className="table">
              <thead>
                <tr>
                  <th>Patient</th>
                  <th>ID</th>
                  <th>ECGs</th>
                  <th>Last ECG</th>
                  <th>Created</th>
                  <th style={{ textAlign: "right" }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(({ p, count, lastEcgAt }) => (
                  <tr key={p.id}>
                    <td>
                      <div style={{ fontWeight: 650 }}>{p.displayName}</div>
                    </td>
                    <td className="mono">{p.patientCode}</td>
                    <td className="mono">{count}</td>
                    <td className="mono">{lastEcgAt ? formatDateTime(lastEcgAt) : "—"}</td>
                    <td className="mono">{formatDateTime(p.createdAt)}</td>
                    <td style={{ textAlign: "right" }}>
                      <div className="row" style={{ justifyContent: "flex-end" }}>
                        <button
                          className="btn"
                          onClick={() => {
                            selectPatient(p.id);
                            onOpenPatient(p.id);
                          }}
                        >
                          Open
                        </button>
                        <button className="btn btnDanger" onClick={() => removePatient(p.id)}>
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      <div style={{ height: 12 }} />

      <div className="panel panelPad">
        <div className="label">New patient</div>
        <div className="subtle">Minimum required: name + ID.</div>
        <div className="row" style={{ marginTop: 10 }}>
          <input className="input" placeholder="Display name (e.g. A. Nguyen)" value={draftName} onChange={(e) => setDraftName(e.target.value)} />
          <input className="input" placeholder="Patient ID (e.g. SS-01942)" value={draftCode} onChange={(e) => setDraftCode(e.target.value)} />
          <button
            className="btn"
            disabled={!draftName.trim() || !draftCode.trim()}
            onClick={async () => {
              const p = await createPatient({ displayName: draftName.trim(), patientCode: draftCode.trim() });
              setDraftName("");
              setDraftCode("");
              selectPatient(p.id);
              onOpenPatient(p.id);
            }}
          >
            Create & open
          </button>
        </div>
      </div>
    </section>
  );
}

function PatientProfileView({ patientId, onBack }: { patientId: string; onBack: () => void }) {
  const { patients, ecgs, addEcgUpload, removeEcgUpload, selectPatient } = usePatients();
  const patient = patients.find((p) => p.id === patientId) ?? null;

  const uploads = useMemo(() => ecgs.filter((e) => e.patientId === patientId).sort((a, b) => b.uploadedAt - a.uploadedAt), [ecgs, patientId]);
  const totalBytes = useMemo(() => uploads.reduce((sum, u) => sum + (u.size ?? 0), 0), [uploads]);

  const [dragOver, setDragOver] = useState(false);
  const [preview, setPreview] = useState<{
    ecgId: string;
    title: string;
    samples: Float32Array;
    fs: number;
  } | null>(null);

  async function openPreview(ecgId: string, title: string, fsHint?: number) {
    const blob = await dbGetEcgBlob(ecgId);
    if (!blob) return;
    const { samples, fs } = await parseEcgBlob(blob);
    setPreview({ ecgId, title, samples, fs: fsHint ?? fs });
  }

  if (!patient) {
    return (
      <section className="panel panelPad">
        <Topbar title="Patient not found" right={<button className="btn" onClick={onBack}>Back</button>} />
        <div className="subtle">This patient no longer exists.</div>
      </section>
    );
  }

  return (
    <>
      <section className="panel panelPad">
        <Topbar
          title={patient.displayName}
          subtitle={`${patient.patientCode} • Created ${formatDateTime(patient.createdAt)}`}
          right={
            <>
              <button className="btn" onClick={() => selectPatient(patient.id)}>
                Set active
              </button>
              <button className="btn" onClick={onBack}>
                Back
              </button>
            </>
          }
        />

        <div className="split">
          <div className="panel panelPad">
            <div className="label">ECG uploads</div>
            <div className="subtle">Supported: CSV/TXT numbers, or JSON array / {"{fs,samples}"}.</div>

            <div className="row" style={{ marginTop: 10 }}>
              <label className="btn" style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                Upload…
                <input
                  type="file"
                  style={{ display: "none" }}
                  multiple
                  accept=".csv,.txt,.json,application/json,text/plain,text/csv"
                  onChange={async (e) => {
                    const files = Array.from(e.target.files ?? []);
                    if (files.length === 0) return;
                    e.currentTarget.value = "";
                    for (const f of files) await addEcgUpload(patient.id, f);
                  }}
                />
              </label>
              <div className="mono">{uploads.length ? `${uploads.length} file(s) • ${formatBytes(totalBytes)}` : "No uploads yet"}</div>
            </div>

            <div style={{ marginTop: 10 }}>
              {uploads.length === 0 ? (
                <div
                  className={["dropZone", dragOver ? "dropZoneActive" : ""].join(" ")}
                  onDragEnter={(e) => {
                    e.preventDefault();
                    setDragOver(true);
                  }}
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragOver(true);
                  }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={async (e) => {
                    e.preventDefault();
                    setDragOver(false);
                    const files = Array.from(e.dataTransfer.files ?? []);
                    for (const f of files) await addEcgUpload(patient.id, f);
                  }}
                >
                  Drop ECG files here
                </div>
              ) : (
                <div className="panel" style={{ background: "var(--panel2)" }}>
                  <div className="scroll" style={{ maxHeight: "46vh" }}>
                    <table className="table">
                      <thead>
                        <tr>
                          <th>Name</th>
                          <th>Uploaded</th>
                          <th>Size</th>
                          <th className="mono">Fs</th>
                          <th style={{ textAlign: "right" }}>Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {uploads.map((u) => (
                          <tr key={u.id}>
                            <td style={{ fontWeight: 650 }}>{u.name}</td>
                            <td className="mono">{formatDateTime(u.uploadedAt)}</td>
                            <td className="mono">{formatBytes(u.size)}</td>
                            <td className="mono">{u.fs ? `${u.fs} Hz` : "—"}</td>
                            <td style={{ textAlign: "right" }}>
                              <div className="row" style={{ justifyContent: "flex-end" }}>
                                <button className="btn" onClick={() => openPreview(u.id, u.name, u.fs)}>
                                  View
                                </button>
                                <button
                                  className="btn"
                                  onClick={async () => {
                                    const blob = await dbGetEcgBlob(u.id);
                                    if (!blob) return;
                                    const url = URL.createObjectURL(blob);
                                    const a = document.createElement("a");
                                    a.href = url;
                                    a.download = u.name;
                                    a.click();
                                    setTimeout(() => URL.revokeObjectURL(url), 2500);
                                  }}
                                >
                                  Download
                                </button>
                                <button className="btn btnDanger" onClick={() => removeEcgUpload(u.id)}>
                                  Delete
                                </button>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="panel panelPad">
            <div className="label">Snapshot</div>
            <div className="subtle" style={{ marginBottom: 10 }}>
              Quick patient-level overview.
            </div>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <div>
                <div className="label">ECGs</div>
                <div className="mono">{uploads.length}</div>
              </div>
              <div>
                <div className="label">Storage</div>
                <div className="mono">{totalBytes ? formatBytes(totalBytes) : "—"}</div>
              </div>
              <div>
                <div className="label">Last</div>
                <div className="mono">{uploads[0]?.uploadedAt ? formatDateTime(uploads[0].uploadedAt) : "—"}</div>
              </div>
            </div>

            <div style={{ height: 12 }} />

            <div className="label">Formats</div>
            <div className="subtle">Uploads are stored locally on this device (IndexedDB).</div>
            <div className="panel" style={{ background: "var(--panel2)", padding: 10, marginTop: 10 }}>
              <div className="mono">{"{ \"fs\": 250, \"samples\": [0.01, 0.02, ...] }"}</div>
            </div>
          </div>
        </div>
      </section>

      {preview ? (
        <div className="modalBackdrop" role="dialog" aria-modal="true" aria-label="ECG preview">
          <div className="modal panel">
            <div className="panelPad">
              <Topbar
                title={preview.title}
                subtitle={`Fs ${preview.fs} Hz • ${preview.samples.length} samples • ${(preview.samples.length / preview.fs).toFixed(1)}s`}
                right={<button className="btn" onClick={() => setPreview(null)}>Close</button>}
              />

              <div className="waveWrap">
                <div className="waveScroll">
                  <Waveform samples={preview.samples} fs={preview.fs} pxPerSecond={26} height={460} />
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}

function AppInner() {
  const [route, setRoute] = useState<Route>({ name: "dashboard" });

  const activeNav = route.name === "dashboard" ? "dashboard" : "patients";

  return (
    <div className="app">
      <Sidebar
        active={activeNav}
        onNavigate={(dest) => setRoute(dest === "dashboard" ? { name: "dashboard" } : { name: "patients" })}
      />

      <main className="main" id="main">
        {route.name === "dashboard" ? <DashboardView /> : null}
        {route.name === "patients" ? (
          <PatientsView
            onOpenPatient={(patientId) => {
              setRoute({ name: "patient", patientId });
            }}
          />
        ) : null}
        {route.name === "patient" ? (
          <PatientProfileView patientId={route.patientId} onBack={() => setRoute({ name: "patients" })} />
        ) : null}
      </main>
    </div>
  );
}

export function App() {
  return (
    <PatientsProvider>
      <AppInner />
    </PatientsProvider>
  );
}


