import React, { useEffect, useMemo, useRef } from "react";

export type Region = {
  id: string;
  t0: number;
  t1: number;
  confidence: number; // 0..1
  reason: string;
};

function downsampleToMax(samples: Float32Array, max: number) {
  if (samples.length <= max) return samples;
  const stride = Math.ceil(samples.length / max);
  const out = new Float32Array(Math.ceil(samples.length / stride));
  for (let i = 0, j = 0; i < samples.length; i += stride, j++) out[j] = samples[i]!;
  return out;
}

function parseNumberList(text: string): number[] {
  const parts = text.trim().split(/[,\s]+/g);
  const out: number[] = [];
  for (const p of parts) {
    const v = Number.parseFloat(p);
    if (Number.isFinite(v)) out.push(v);
  }
  return out;
}

export async function parseEcgBlob(blob: Blob): Promise<{ samples: Float32Array; fs: number }> {
  const text = await blob.text();
  const trimmed = text.trim();

  let fs = 250;
  let values: number[] = [];

  if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
    try {
      const parsed = JSON.parse(trimmed) as unknown;
      if (Array.isArray(parsed)) {
        values = parsed.map((x) => Number(x)).filter((x) => Number.isFinite(x));
      } else if (parsed && typeof parsed === "object") {
        const p = parsed as Record<string, unknown>;
        const maybeFs = p.fs ?? p.samplingRate ?? p.sampleRate;
        const maybeSamples = p.samples ?? p.data ?? p.values;
        if (typeof maybeFs === "number" && Number.isFinite(maybeFs)) fs = maybeFs;
        if (Array.isArray(maybeSamples)) {
          values = maybeSamples.map((x) => Number(x)).filter((x) => Number.isFinite(x));
        } else if (typeof maybeSamples === "string") {
          values = parseNumberList(maybeSamples);
        }
      }
    } catch {
      values = parseNumberList(trimmed);
    }
  } else {
    values = parseNumberList(trimmed);
  }

  const samples = downsampleToMax(Float32Array.from(values), 60_000);
  return { samples, fs };
}

export function synthEcg({ seconds, fs, seed = 1 }: { seconds: number; fs: number; seed?: number }) {
  // Simple synthetic lead: baseline wander + pseudo-QRS spikes + noise.
  const n = Math.max(1, Math.floor(seconds * fs));
  const out = new Float32Array(n);
  let s = seed >>> 0;
  const rand = () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return (s & 0xfffffff) / 0xfffffff;
  };

  const hr = 62 + rand() * 12;
  const beatPeriod = 60 / hr;

  for (let i = 0; i < n; i++) {
    const t = i / fs;
    const phase = (t % beatPeriod) / beatPeriod;
    const baseline = 0.04 * Math.sin(t * 2 * Math.PI * 0.28);
    const qrs = Math.exp(-Math.pow((phase - 0.18) / 0.012, 2)) * 1.0 - Math.exp(-Math.pow((phase - 0.205) / 0.018, 2)) * 0.25;
    const p = Math.exp(-Math.pow((phase - 0.07) / 0.03, 2)) * 0.12;
    const tw = Math.exp(-Math.pow((phase - 0.42) / 0.08, 2)) * 0.25;
    const noise = (rand() - 0.5) * 0.05;
    out[i] = baseline + p + qrs + tw + noise;
  }
  return out;
}

function computeMinMax(samples: Float32Array) {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < samples.length; i++) {
    const v = samples[i]!;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) return { min: -1, max: 1 };
  return { min, max };
}

export function Waveform({
  samples,
  fs,
  pxPerSecond,
  height,
  regions = [],
  onHoverRegion,
  onLeaveRegion,
  onClickRegion
}: {
  samples: Float32Array;
  fs: number;
  pxPerSecond: number;
  height: number;
  regions?: Region[];
  onHoverRegion?: (r: Region, clientX: number, clientY: number) => void;
  onLeaveRegion?: () => void;
  onClickRegion?: (r: Region) => void;
}) {
  const secondsTotal = Math.max(1, samples.length / fs);
  const width = Math.max(480, Math.round(secondsTotal * pxPerSecond));

  const down = useMemo(() => {
    // Approx 1 point per pixel (good enough for preview & overview)
    const max = Math.max(800, width);
    return downsampleToMax(samples, max);
  }, [samples, width]);

  const { min, max } = useMemo(() => computeMinMax(down), [down]);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.floor(width * dpr);
    c.height = Math.floor(height * dpr);
    c.style.width = `${width}px`;
    c.style.height = `${height}px`;

    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    // Background
    ctx.fillStyle = "rgba(0,0,0,0.45)";
    ctx.fillRect(0, 0, width, height);

    // Light grid
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    const step = 24;
    for (let x = 0; x <= width; x += step) {
      ctx.beginPath();
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, height);
      ctx.stroke();
    }
    for (let y = 0; y <= height; y += step) {
      ctx.beginPath();
      ctx.moveTo(0, y + 0.5);
      ctx.lineTo(width, y + 0.5);
      ctx.stroke();
    }

    // Trace
    const range = max - min;
    const padY = 10;
    const yScale = (height - padY * 2) / range;
    const xScale = width / Math.max(1, down.length - 1);

    ctx.strokeStyle = "rgba(80,255,160,0.9)";
    ctx.lineWidth = 1.5;
    ctx.shadowColor = "rgba(80,255,160,0.25)";
    ctx.shadowBlur = 10;

    ctx.beginPath();
    for (let i = 0; i < down.length; i++) {
      const x = i * xScale;
      const v = down[i]!;
      const y = height - padY - (v - min) * yScale;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }, [down, width, height, min, max]);

  return (
    <div style={{ position: "relative", display: "inline-block" }}>
      <canvas ref={canvasRef} />

      {/* Clickable regions (simple overlays) */}
      {regions.map((r) => {
        const left = (r.t0 / secondsTotal) * width;
        const right = (r.t1 / secondsTotal) * width;
        const w = Math.max(4, right - left);
        return (
          <div
            key={r.id}
            style={{
              position: "absolute",
              left,
              top: 0,
              width: w,
              height,
              background: "rgba(35,255,240,0.07)",
              borderLeft: "1px solid rgba(35,255,240,0.22)",
              borderRight: "1px solid rgba(35,255,240,0.22)",
              cursor: onClickRegion ? "pointer" : "default"
            }}
            onMouseMove={(e) => onHoverRegion?.(r, e.clientX, e.clientY)}
            onMouseLeave={() => onLeaveRegion?.()}
            onClick={() => onClickRegion?.(r)}
            aria-label={`Region ${r.t0.toFixed(1)}s to ${r.t1.toFixed(1)}s`}
            role={onClickRegion ? "button" : undefined}
            tabIndex={-1}
          />
        );
      })}
    </div>
  );
}


