# AgenticCardioGram Landing Page - Implementation Summary

## Overview
A beautiful, Cluely-inspired landing page for the AgenticCardioGram research project - a machine learning-powered ECG analysis system for sleep apnea classification.

## Tech Stack
- **Framework**: React 19.2.0 with TypeScript
- **Routing**: React Router DOM 7.13.0
- **Styling**: CSS Modules (no external dependencies required)
- **Build Tool**: Vite 7.2.4

## File Structure

```
src/
├── pages/
│   ├── NewLandingPage.tsx          # Main landing page container
│   ├── LandingPage.tsx             # Original landing page (preserved at /old)
│   └── PatientAnalysis.tsx         # Analysis dashboard
├── components/
│   └── landing/
│       ├── Hero.tsx                # Hero section with headline & product mock
│       ├── Stats.tsx               # Key statistics banner
│       ├── Problem.tsx             # Problem statement section
│       ├── Architecture.tsx        # System architecture pipeline
│       ├── Model.tsx               # Model details & metrics
│       ├── Explainability.tsx      # Explainability features
│       ├── Demo.tsx                # Interactive demo with tabs
│       ├── FAQ.tsx                 # Accordion FAQ
│       └── Footer.tsx              # Footer with links & disclaimer
└── styles/
    ├── NewLandingPage.module.css   # Main page styles
    ├── Hero.module.css             # Hero section styles
    ├── Stats.module.css            # Stats section styles
    ├── Problem.module.css          # Problem section styles
    ├── Architecture.module.css     # Architecture section styles
    ├── Model.module.css            # Model section styles
    ├── Explainability.module.css   # Explainability section styles
    ├── Demo.module.css             # Demo section styles
    ├── FAQ.module.css              # FAQ section styles
    └── Footer.module.css           # Footer styles
```

## Design Features

### Visual Aesthetic (Cluely-inspired)
- ✅ Light, airy background with subtle blue/purple gradients
- ✅ Animated blur blobs for depth
- ✅ Sticky translucent navigation
- ✅ Generous whitespace
- ✅ Rounded-2xl cards with soft shadows
- ✅ Tight typography hierarchy
- ✅ Smooth hover transitions

### Navigation
- Sticky navbar with blur backdrop on scroll
- Links: How it works, Research, Team, GitHub
- Logo with ECG waveform icon

### Sections (in order)

#### 1. Hero
- **Headline**: "Sleep Apnea Screening From Single-Lead ECG"
- **Subheadline**: ML-powered analysis with explainable AI
- **CTAs**: 
  - Primary: "View Demo" (navigates to /analysis)
  - Secondary: "Read the Paper" (anchor link)
- **Badges**: Single-lead ECG, CNN-Transformer, Explainability Agent
- **Product Mock**: Interactive card showing:
  - ECG signal visualization
  - Risk timeline (color-coded segments)
  - Metrics (Overall Risk, Confidence, Apnea Windows)

#### 2. Stats
- Built at Western AI / Western University
- PhysioNet Apnea-ECG Dataset
- Key metrics: 87.3% accuracy, 5-min windows, 2-channel input

#### 3. Why This Matters (Problem)
- PSG limitations: cost, wait times, scalability
- 3 problem cards: months-long waits, high cost, scalability limits
- Solution box: ECG-only screening for intelligent triage

#### 4. Architecture
- 3-step pipeline visualization:
  1. Preprocess ECG (R-peak detection)
  2. Build 5-min windows (RRI + R-peak amplitude)
  3. Two-channel input to model

#### 5. Model Details
- Hybrid CNN-Transformer architecture diagram
- Performance metrics grid:
  - 87.3% Accuracy
  - 85.1% Sensitivity
  - 89.4% Specificity
  - 0.92 AUC-ROC
- Highlights: patient-level evaluation, cross-validation

#### 6. Explainability
- Two technique cards:
  1. **HRV Feature Summaries**: VLF/HF ratio, sample entropy, RMSSD
  2. **Grad-CAM-style Highlighting**: Visual attention heatmap
- Benefits: trust, validation, debugging, regulatory compliance

#### 7. Interactive Demo
- 3 tabs: ECG Signal, Risk Timeline, Explanations
- **ECG Tab**: Full overnight ECG with highlighted apnea regions
- **Risk Tab**: Per-segment classification bar chart, overall score
- **Explanation Tab**: Key findings (HRV biomarkers) + clinical recommendation
- CTA: "Try the Full Web App"

#### 8. FAQ (Accordion)
- Is this a replacement for PSG?
- What kind of ECG data?
- How accurate is the model?
- Can I use this clinically?
- System requirements?
- How does explainability work?

#### 9. Footer
- Brand + tagline
- Links: Research, Resources, Team
- Medical disclaimer (prominent)
- Built at Western AI / Western University
- Social links (GitHub, Twitter)

## Key Features

### Responsive Design
- Desktop: Full grid layouts
- Tablet: Adjusted columns
- Mobile: Single-column stacks

### Accessibility
- Semantic HTML
- ARIA labels where needed
- Keyboard navigation support
- Focus states

### Animations
- Floating blur blobs
- Smooth hover transitions
- Card lift effects
- Accordion expand/collapse

### Interactive Elements
- Tabbed demo interface
- Accordion FAQ
- Sticky navigation with blur
- Hover states on all interactive elements

## Content Strategy

### Tone
- Professional but approachable
- Clear, concise copy (SaaS-style)
- Technically accurate
- Research-focused but accessible

### Key Messages
1. Single-lead ECG screening (accessible)
2. ML-powered (CNN-Transformer)
3. Explainable AI (clinical trust)
4. Screening tool, not diagnostic device
5. Research prototype from Western University

## Routes

- `/` - New Cluely-inspired landing page
- `/old` - Original landing page (preserved)
- `/analysis` - Patient analysis dashboard

## Development

### Running the Project
```bash
cd src/ui/ecg-triage-ui
npm install
npm run dev
```

### Building for Production
```bash
npm run build
npm run preview
```

## Customization Points

### Easy to Update
1. **Links**: Update GitHub, paper, contact links in Hero and Footer
2. **Metrics**: Adjust performance numbers in Stats and Model sections
3. **Content**: All copy is in component files, easy to find and edit
4. **Colors**: Main brand colors are in CSS variables (can add to :root)
5. **Assets**: Replace placeholder SVGs with real logos/graphics

### Placeholder Links
- GitHub: `https://github.com`
- Paper: `#research` anchor
- Demo: navigates to `/analysis`
- Contact: `mailto:contact@example.com`

## Browser Compatibility
- Modern browsers (Chrome, Firefox, Safari, Edge)
- CSS Grid & Flexbox
- CSS backdrop-filter for blur effects
- SVG support

## Performance
- No external CSS frameworks (fast load)
- CSS Modules (scoped, no conflicts)
- Code-split by route
- Optimized SVG graphics
- Lazy loading compatible

## Accessibility Checklist
- ✅ Semantic HTML elements
- ✅ Color contrast ratios (WCAG AA)
- ✅ Focus indicators
- ✅ Keyboard navigation
- ✅ Readable font sizes (16px minimum)
- ✅ Descriptive link text
- ✅ Icon alternatives

## Medical Disclaimer
Prominently displayed in footer and throughout:
- This is a research prototype
- Screening tool, not diagnostic device
- Requires PSG confirmation
- Not cleared by regulatory bodies
- For educational/research purposes only

## Design Credits
- Inspired by Cluely's clean, modern aesthetic
- Original content and assets
- No copied text or branding

## Next Steps
1. ✅ Replace placeholder links with real URLs
2. ✅ Add real GitHub repository link
3. ✅ Link to published paper when available
4. ✅ Add team member information
5. ✅ Integrate real demo data
6. ✅ Add analytics tracking (optional)
7. ✅ SEO optimization (meta tags, etc.)

---

**Built by**: Senior Product Designer + Frontend Engineer  
**Framework**: React + TypeScript + CSS Modules  
**Design Inspiration**: Cluely (layout/aesthetic only)  
**Content**: Original, research-based

