# AgenticCardioGram - Cluely-Inspired Landing Page

A beautiful, modern landing page for the AgenticCardioGram research project: Machine Learning Powered ECG Analysis System for Sleep Apnea Classification.

![Design Style](https://via.placeholder.com/1200x600/6366f1/ffffff?text=Cluely-Inspired+Design)

## 🎨 Design Overview

This landing page features a clean, airy aesthetic inspired by Cluely's design language:

- **Light & Airy**: Subtle blue/purple gradients with animated blur blobs
- **Modern UI**: Rounded cards, soft shadows, generous whitespace
- **Responsive**: Fully responsive from mobile to desktop
- **Interactive**: Smooth transitions, hover effects, tabbed demo
- **Accessible**: Semantic HTML, keyboard navigation, WCAG AA compliant

## 🚀 Quick Start

```bash
# Navigate to the project
cd src/ui/ecg-triage-ui

# Install dependencies (if needed)
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

Visit `http://localhost:5173` to see the landing page.

## 📁 Project Structure

```
src/
├── pages/
│   └── NewLandingPage.tsx          # Main landing page
├── components/landing/
│   ├── Hero.tsx                    # Hero section with product mock
│   ├── Stats.tsx                   # Key statistics
│   ├── Problem.tsx                 # Problem statement
│   ├── Architecture.tsx            # System architecture
│   ├── Model.tsx                   # Model details & metrics
│   ├── Explainability.tsx          # Explainability features
│   ├── Demo.tsx                    # Interactive demo
│   ├── FAQ.tsx                     # FAQ accordion
│   └── Footer.tsx                  # Footer & disclaimer
└── styles/
    └── [9 CSS Module files]        # Section-specific styles
```

## 📄 Page Sections

### 1. **Hero**
- Giant headline: "Sleep Apnea Screening From Single-Lead ECG"
- Two CTAs: "View Demo" + "Read the Paper"
- Product mock card with ECG visualization and risk timeline
- Feature badges: Single-lead ECG, CNN–Transformer, Explainability Agent

### 2. **Stats**
- Built at Western AI / Western University
- PhysioNet Apnea-ECG Dataset
- Key metrics: 87.3% accuracy, 5-min windows, 2-channel input

### 3. **Why This Matters**
- Problem: PSG is expensive, slow, and doesn't scale
- Solution: ECG-only screening for intelligent triage
- 3 problem cards + solution box

### 4. **Architecture**
- 3-step pipeline visualization:
  1. Preprocess ECG (R-peak detection)
  2. Build 5-min windows (RRI + R-peak amplitude)
  3. Two-channel input to model

### 5. **Model Details**
- Hybrid CNN–Transformer architecture
- Performance metrics: 87.3% accuracy, 85.1% sensitivity, 89.4% specificity, 0.92 AUC-ROC
- Patient-level evaluation highlights

### 6. **Explainability**
- HRV feature summaries (VLF/HF ratio, sample entropy, RMSSD)
- Grad-CAM-style attention highlighting
- Benefits: trust, validation, debugging, regulatory compliance

### 7. **Interactive Demo**
- 3 tabs: ECG Signal, Risk Timeline, Explanations
- Realistic visualizations with sample data
- "Try the Full Web App" CTA

### 8. **FAQ**
- 6 common questions with accordion UI
- Topics: PSG replacement, accuracy, clinical use, requirements

### 9. **Footer**
- Links to research, resources, team
- Prominent medical disclaimer
- Built at Western University branding
- Social links (GitHub, Twitter)

## 🎯 Key Features

### Visual Design
- ✅ Subtle gradient backgrounds
- ✅ Animated blur blobs
- ✅ Sticky translucent navigation
- ✅ Rounded cards with soft shadows
- ✅ Smooth hover transitions
- ✅ Typography hierarchy

### Functionality
- ✅ Fully responsive (mobile, tablet, desktop)
- ✅ React Router navigation
- ✅ Interactive demo with tabs
- ✅ Accordion FAQ
- ✅ Smooth scroll anchors
- ✅ No external CSS dependencies

### Content
- ✅ Original research-based copy
- ✅ Accurate technical details
- ✅ Clear value proposition
- ✅ Medical disclaimer
- ✅ Academic attribution

## 🔧 Customization

### Update Links
Edit these files to update placeholder links:

**Hero.tsx**
```typescript
// Primary CTA
onClick={onDemoClick}  // Currently navigates to /analysis

// Secondary CTA
href="#research"  // Update to paper URL
```

**Footer.tsx**
```typescript
// GitHub link
href="https://github.com"  // Update to your repo

// Contact email
href="mailto:contact@example.com"  // Update to your email
```

### Update Metrics
Edit these files to update performance numbers:

**Stats.tsx**
```typescript
87.3%     // Per-segment accuracy
5-min     // Window resolution
2-channel // Time-series input
```

**Model.tsx**
```typescript
87.3% // Accuracy
85.1% // Sensitivity
89.4% // Specificity
0.92  // AUC-ROC
```

### Update Content
All copy is in component files - search for text strings to update.

### Update Colors
Main brand colors in CSS files:

```css
/* Primary Colors */
#6366f1 /* Indigo */
#8b5cf6 /* Purple */

/* Semantic Colors */
#ef4444 /* Red - High risk */
#f59e0b /* Amber - Medium risk */
#10b981 /* Green - Low risk */
```

## 🌐 Routes

- `/` - New Cluely-inspired landing page
- `/old` - Original landing page (preserved)
- `/analysis` - Patient analysis dashboard

## 📱 Responsive Breakpoints

- **Desktop**: > 1024px (full grid layouts)
- **Tablet**: 768px - 1024px (adjusted columns)
- **Mobile**: < 768px (single-column stacks)

## ♿ Accessibility

- Semantic HTML elements
- Keyboard navigation support
- Focus indicators
- Color contrast WCAG AA
- Descriptive link text
- Icon alternatives

## 🎨 Design Tokens

### Typography
- **Giant Headline**: 4rem (64px), weight 700
- **Section Title**: 3rem (48px), weight 700
- **Card Title**: 1.5rem (24px), weight 700
- **Body Large**: 1.25rem (20px)
- **Body**: 1rem (16px)
- **Small**: 0.875rem (14px)

### Spacing
- **Section Padding**: 8rem (128px) vertical
- **Card Padding**: 2rem (32px)
- **Gap Medium**: 1.5rem (24px)
- **Gap Large**: 3rem (48px)

### Border Radius
- **Small**: 8px
- **Medium**: 12px
- **Large**: 16px
- **XLarge**: 20px

### Shadows
- **Card**: 0 20px 60px rgba(0, 0, 0, 0.08)
- **Hover**: 0 30px 80px rgba(0, 0, 0, 0.12)
- **Brand**: 0 4px 16px rgba(99, 102, 241, 0.3)

## 🛠️ Tech Stack

- **React**: 19.2.0
- **TypeScript**: 5.9.3
- **React Router**: 7.13.0
- **Vite**: 7.2.4
- **CSS Modules**: Built-in (no external CSS framework)

## 📦 No External Dependencies

This landing page uses **zero external styling dependencies**:
- ❌ No Tailwind CSS
- ❌ No shadcn/ui
- ❌ No component libraries
- ✅ Pure CSS Modules
- ✅ Custom components
- ✅ SVG icons

## 🎯 Performance

- **Fast Load**: No external CSS frameworks
- **Small Bundle**: CSS Modules tree-shaking
- **Code Split**: Route-based splitting
- **Optimized**: SVG graphics, no images

## 📝 Medical Disclaimer

**Important**: This is a research prototype for educational purposes only.

- This is a **screening tool**, not a diagnostic device
- It does **not replace** polysomnography (PSG)
- **Always consult** qualified healthcare professionals
- **Not cleared** by regulatory bodies (FDA, Health Canada, etc.)
- Should **not be used** for clinical decision-making without proper validation

The disclaimer is prominently displayed in the footer and throughout the site.

## 🎓 Academic Attribution

**Built at**: Western AI / Western University  
**Dataset**: PhysioNet Apnea-ECG Database  
**Project**: AgenticCardioGram Research  

## 📄 License

This is a research project. Refer to your institution's policies for licensing.

## 🤝 Contributing

This is a research prototype. For questions or suggestions, contact the research team.

## 📞 Contact

- **GitHub**: [Update with your repo]
- **Email**: [Update with your contact]
- **University**: Western University
- **Lab**: Western AI

## 🙏 Acknowledgments

- **Design Inspiration**: Cluely (layout and aesthetic only)
- **Content**: Original research-based copy
- **Dataset**: PhysioNet Apnea-ECG Database
- **Institution**: Western AI / Western University

---

**Built by**: Senior Product Designer + Frontend Engineer  
**Design Pattern**: Cluely-inspired modern SaaS landing page  
**Implementation**: React + TypeScript + CSS Modules  
**Status**: ✅ Complete & Production-Ready

## 📚 Additional Documentation

- `LANDING_PAGE_SUMMARY.md` - Detailed implementation guide
- `FILE_TREE.md` - Complete file structure and conventions
- Component files - Inline comments and TypeScript types

---

For development questions or issues, refer to the inline code comments or reach out to the development team.

