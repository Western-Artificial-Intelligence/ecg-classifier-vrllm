# 🚀 Quick Start Guide

## Start the Application (2 commands)

### 1. Start Frontend
```bash
cd src/ui/ecg-triage-ui
npm run dev
```
✅ Opens at `http://localhost:5173` (or next available port)

### 2. Start Backend (if needed)
```bash
cd src/backend
python main.py
```
✅ Runs at `http://localhost:8000`

---

## What You'll See

### Landing Page (`/`)
- Beautiful hero section
- Feature showcase
- Workflow explanation
- "Start Analysis" button → takes you to the app

### Analysis Page (`/analysis`)
- Upload ECG files (.dat format)
- View ECG waveforms (3 modes, 4 zoom levels)
- Run AI predictions
- See Grad-CAM explanations
- Chat with AI assistant

---

## Quick Demo Flow

1. **Visit** `http://localhost:5173`
2. **Click** "Start Analysis" or "Launch App"
3. **Select** an existing ECG file (or upload new one)
4. **Click** "Run Prediction" (needs backend running)
5. **Review** flagged apnea segments
6. **Click** "Explain" to see Grad-CAM heatmaps
7. **Ask** AI assistant questions about the analysis

---

## Keyboard Shortcuts (Analysis Page)

| Key | Action |
|-----|--------|
| `1` | Detail view (1x zoom) |
| `5` | 1-minute view (5x zoom) |
| `0` | 5-minute view (10x zoom) |
| `F` | Full recording view |
| `W` | Waveform mode |
| `M` | Minimap mode |
| `S` | Summary mode |

---

## Pages

- **Landing**: `/` - Marketing/intro page
- **Analysis**: `/analysis` - Main ECG analysis tool

---

## File Locations

- **Frontend Code**: `src/ui/ecg-triage-ui/src/`
- **Landing Page**: `src/ui/ecg-triage-ui/src/pages/LandingPage.tsx`
- **Analysis Page**: `src/ui/ecg-triage-ui/src/pages/PatientAnalysis.tsx`
- **Styles**: `src/ui/ecg-triage-ui/src/styles/`

---

## Troubleshooting

### "Port in use" error?
Vite will auto-select next available port (5174, 5175, etc.)

### Backend not connecting?
Make sure backend is running: `python src/backend/main.py`

### Dependencies missing?
Run: `cd src/ui/ecg-triage-ui && npm install`

---

## Documentation

- 📘 Full README: `src/ui/ecg-triage-ui/README.md`
- 🎨 UI Overview: `UI_OVERVIEW.md`
- ⚙️ Setup Guide: `FRONTEND_SETUP.md`
- 🚀 Deployment: `DEPLOYMENT_SUMMARY.md`

---

**That's it! You're ready to go. 🎉**

