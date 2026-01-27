# Frontend Setup Guide

## Quick Start

### 1. Navigate to the UI directory

```bash
cd src/ui/ecg-triage-ui
```

### 2. Install dependencies

```bash
npm install
```

### 3. Start the development server

```bash
npm run dev
```

The application will be available at `http://localhost:5173` (or next available port).

## Application Routes

- `/` - Landing page with project overview and features
- `/analysis` - Patient ECG analysis interface

## Development Workflow

### Running the Frontend Only

```bash
cd src/ui/ecg-triage-ui
npm run dev
```

### Running with Backend

Terminal 1 (Backend):
```bash
cd src/backend
python main.py
```

Terminal 2 (Frontend):
```bash
cd src/ui/ecg-triage-ui
npm run dev
```

The frontend will automatically proxy API requests to `http://localhost:8000`.

## Building for Production

```bash
npm run build
```

Built files will be in the `dist/` directory.

## Project Structure

```
ecg-triage-ui/
├── src/
│   ├── pages/              # Route pages
│   │   ├── LandingPage.tsx
│   │   └── PatientAnalysis.tsx
│   ├── components/         # Reusable components
│   ├── styles/             # CSS modules
│   ├── App.tsx            # Router setup
│   └── main.tsx           # Entry point
├── public/                 # Static assets
├── package.json
└── vite.config.ts
```

## Key Features

### Landing Page
- Modern clinical design
- Feature showcase
- Clear CTAs
- Professional disclaimers

### Patient Analysis
- ECG waveform viewer (3 view modes)
- File management
- AI predictions with Grad-CAM
- Interactive chat assistant

## Environment Configuration

No environment variables needed for development. The frontend uses:
- Backend API: `http://localhost:8000`
- Frontend Dev Server: `http://localhost:5173`

## Troubleshooting

### Port Already in Use

If port 5173 is occupied, Vite will automatically try the next available port (5174, 5175, etc.).

### Backend Connection Issues

Make sure the backend is running on `http://localhost:8000`. Check:
```bash
curl http://localhost:8000/
```

### Missing Dependencies

```bash
rm -rf node_modules package-lock.json
npm install
```

## Design System

The UI uses a clinical biotech aesthetic with:
- **Primary Color**: Indigo/Violet gradient (`#6366f1` → `#8b5cf6`)
- **Font**: Inter (via Google Fonts)
- **Layout**: Responsive with CSS Grid/Flexbox
- **Components**: Custom-styled with CSS Modules

## Performance

- Vite provides instant HMR (Hot Module Replacement)
- Chart.js handles large ECG datasets efficiently
- React 19 with concurrent features for smooth UI

## Browser Support

- Chrome/Edge: Latest 2 versions
- Firefox: Latest 2 versions
- Safari: 14+

---

**Ready to start? Run `npm run dev` and visit http://localhost:5173**

