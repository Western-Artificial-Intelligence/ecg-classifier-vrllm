# ApneaScreen - AI-Powered Sleep Apnea Screening Tool

A modern, clinical-grade desktop MVP for AI-powered sleep apnea screening using single-lead ECG data.

## 🎯 Project Overview

ApneaScreen is a clinical screening assistant that analyzes overnight ECG recordings to identify potential sleep apnea events. It provides:

- **Risk Stratification**: Categorizes patients into Low/Medium/High risk levels
- **Segment Highlighting**: Pinpoints exact time windows with suspected apnea events
- **Explainable AI**: Shows interpretable biomarkers (HRV, entropy) behind predictions
- **Conversational Assistant**: AI-powered chat for analysis interpretation and next steps

### What It Is NOT
- Not a medical device or diagnostic tool
- Not for real-time monitoring
- Not multi-signal (ECG only)
- Not FDA-approved

## 🚀 Features

### Landing Page
- Modern, clinical aesthetic inspired by premium biotech design
- Clear value proposition and feature showcase
- Workflow explanation (Upload → Analyze → Review → Act)
- Professional trust indicators and disclaimers

### Patient Analysis Page
- **ECG Waveform Viewer**: Interactive visualization with multiple zoom levels
- **Minimap View**: Overview of entire recording with risk highlighting
- **Summary Charts**: Bar charts showing apnea probability by minute
- **Grad-CAM Explainability**: Visual explanations of model predictions
- **AI Assistant**: Conversational interface for questions and guidance
- **File Management**: Upload and manage multiple ECG recordings

## 🛠️ Tech Stack

- **Frontend**: React 19 + TypeScript
- **Build Tool**: Vite
- **Routing**: React Router v6
- **Charting**: Chart.js + react-chartjs-2
- **Styling**: CSS Modules
- **Backend API**: Python FastAPI (separate service)

## 📦 Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## 🎨 Design System

### Colors
- **Primary**: `#6366f1` (Indigo 500) → `#8b5cf6` (Violet 500) gradient
- **Success**: `#10b981` (Emerald 500)
- **Danger**: `#d32f2f` (Red 700)
- **Background**: `#f8f9fe` → `#ffffff` gradient
- **Text**: `#1a1a2e` (Primary), `#4a5568` (Secondary)

### Typography
- **Font**: Inter (Google Fonts)
- **Weights**: 400 (Regular), 500 (Medium), 600 (Semi-bold), 700 (Bold), 800 (Extra-bold)

## 📂 Project Structure

```
src/
├── pages/
│   ├── LandingPage.tsx          # Marketing/intro page
│   └── PatientAnalysis.tsx      # Main ECG analysis interface
├── components/
│   ├── EcgChart.tsx             # Waveform visualization
│   ├── MinimapView.tsx          # Full recording overview
│   └── SummaryChartView.tsx     # Risk probability charts
├── styles/
│   ├── LandingPage.module.css   # Landing page styles
│   └── App.module.css           # Analysis page styles
├── App.tsx                       # Router setup
├── main.tsx                      # Entry point
└── index.css                     # Global styles
```

## 🔌 Backend Integration

The frontend expects a FastAPI backend running on `http://localhost:8000` with these endpoints:

- `GET /api/ecg_data/{filename}` - Fetch ECG signal data
- `POST /api/predict/{record_name}` - Run apnea prediction
- `POST /api/gradcam/{record_name}?minute={N}` - Generate Grad-CAM visualization

See backend documentation for setup instructions.

## 🎯 Usage

### Starting the Application

1. Launch the app (navigates to Landing Page)
2. Click "Start Analysis" or "Launch App" to enter the analysis interface
3. Upload ECG data files (.dat format)
4. Select a file to view waveform
5. Click "Run Prediction" to analyze for apnea events
6. Review flagged segments and use "Explain" to see Grad-CAM visualizations
7. Use AI Assistant for interpretation and next steps

### Keyboard Shortcuts (Analysis Page)

- `1` - Detail view (1x zoom)
- `5` - 1-minute view (5x zoom)
- `0` - 5-minute view (10x zoom)
- `F` - Full recording view
- `W` - Waveform mode
- `M` - Minimap mode
- `S` - Summary mode

## 🏥 Clinical Context

### Target Users
- Clinicians conducting preliminary sleep apnea screening
- Sleep researchers analyzing ECG data
- Health-tech evaluators and investors (CUCAI competition judges)

### Workflow Integration
1. **Screen**: Use ApneaScreen to identify high-risk patients
2. **Prioritize**: Focus clinical resources on flagged cases
3. **Refer**: Send high-risk patients for polysomnography (PSG)
4. **Document**: Export analysis results for clinical records

## ⚠️ Medical Disclaimer

**This tool is for screening purposes only and does not diagnose sleep apnea.**

Always consult with qualified healthcare professionals for medical decisions. This is a demonstration MVP built for the CUCAI AI Competition and is not approved for clinical use.

## 📊 Model Performance

The underlying AI model achieves:
- **Sensitivity**: ~94% (detecting true apnea events)
- **Specificity**: ~92% (avoiding false positives)
- **AUC-ROC**: 0.96

*Metrics based on MIT-BIH Apnea-ECG database evaluation*

## 🚧 Known Limitations

- Single-lead ECG only (no multi-channel support)
- Requires overnight recordings (6-8 hours typical)
- Not validated for pediatric cases
- May underperform on patients with arrhythmias
- Requires stable ECG signal quality

## 🔮 Future Enhancements

- [ ] Multi-lead ECG support
- [ ] Real-time streaming analysis
- [ ] Integration with EHR systems
- [ ] Mobile responsive design
- [ ] PDF report generation
- [ ] Batch processing for multiple patients
- [ ] Advanced filtering and noise reduction
- [ ] Multi-language support

## 📝 License

This project is built for educational and demonstration purposes as part of the CUCAI Competition.

## 🙏 Acknowledgments

- Design inspiration: [Arooth.webflow.io](https://arooth.webflow.io/)
- Dataset: MIT-BIH Apnea-ECG Database
- Competition: CUCAI (Canadian Undergraduate Conference on AI)

## 📧 Contact

For questions about this project, please reach out through the competition platform.

---

**Built with ❤️ for better sleep health screening**
