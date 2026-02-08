# AgenticCardioGram Landing Page - Complete File Tree

```
ecg-triage-ui/
│
├── package.json                           # Dependencies (React, React Router, Vite)
├── vite.config.ts                        # Vite configuration
├── tsconfig.json                         # TypeScript configuration
│
├── public/                               # Static assets
│
├── src/
│   ├── main.tsx                          # App entry point
│   ├── App.tsx                           # ✨ UPDATED: Routes configuration
│   ├── index.css                         # Global styles
│   │
│   ├── pages/
│   │   ├── NewLandingPage.tsx           # 🆕 Main landing page (Cluely-inspired)
│   │   ├── LandingPage.tsx              # Original landing page (preserved)
│   │   ├── PatientAnalysis.tsx          # Analysis dashboard
│   │   └── PatientDashboard.tsx         # Patient dashboard
│   │
│   ├── components/
│   │   ├── landing/                      # 🆕 Landing page components
│   │   │   ├── Hero.tsx                 # Hero section with product mock
│   │   │   ├── Stats.tsx                # Statistics banner
│   │   │   ├── Problem.tsx              # Problem statement
│   │   │   ├── Architecture.tsx         # System architecture
│   │   │   ├── Model.tsx                # Model details & metrics
│   │   │   ├── Explainability.tsx       # Explainability features
│   │   │   ├── Demo.tsx                 # Interactive demo with tabs
│   │   │   ├── FAQ.tsx                  # Accordion FAQ
│   │   │   └── Footer.tsx               # Footer with disclaimer
│   │   │
│   │   ├── EcgChart.tsx                 # ECG visualization
│   │   ├── MinimapView.tsx              # Minimap component
│   │   └── SummaryChartView.tsx         # Summary charts
│   │
│   ├── styles/
│   │   ├── NewLandingPage.module.css    # 🆕 Main page & navigation styles
│   │   ├── Hero.module.css              # 🆕 Hero section styles
│   │   ├── Stats.module.css             # 🆕 Stats section styles
│   │   ├── Problem.module.css           # 🆕 Problem section styles
│   │   ├── Architecture.module.css      # 🆕 Architecture section styles
│   │   ├── Model.module.css             # 🆕 Model section styles
│   │   ├── Explainability.module.css    # 🆕 Explainability section styles
│   │   ├── Demo.module.css              # 🆕 Demo section styles
│   │   ├── FAQ.module.css               # 🆕 FAQ section styles
│   │   ├── Footer.module.css            # 🆕 Footer styles
│   │   ├── LandingPage.module.css       # Original landing page styles
│   │   ├── PatientDashboard.module.css  # Dashboard styles
│   │   └── App.module.css               # App-level styles
│   │
│   └── assets/
│       └── react.svg                     # React logo
│
└── LANDING_PAGE_SUMMARY.md               # 🆕 Implementation documentation

```

## New Files Created (17 files)

### Components (9 files)
1. `/src/pages/NewLandingPage.tsx` - Main landing page container
2. `/src/components/landing/Hero.tsx` - Hero section
3. `/src/components/landing/Stats.tsx` - Statistics
4. `/src/components/landing/Problem.tsx` - Problem statement
5. `/src/components/landing/Architecture.tsx` - Architecture pipeline
6. `/src/components/landing/Model.tsx` - Model details
7. `/src/components/landing/Explainability.tsx` - Explainability
8. `/src/components/landing/Demo.tsx` - Interactive demo
9. `/src/components/landing/FAQ.tsx` - FAQ accordion
10. `/src/components/landing/Footer.tsx` - Footer

### Styles (9 files)
1. `/src/styles/NewLandingPage.module.css` - Main page styles
2. `/src/styles/Hero.module.css` - Hero styles
3. `/src/styles/Stats.module.css` - Stats styles
4. `/src/styles/Problem.module.css` - Problem styles
5. `/src/styles/Architecture.module.css` - Architecture styles
6. `/src/styles/Model.module.css` - Model styles
7. `/src/styles/Explainability.module.css` - Explainability styles
8. `/src/styles/Demo.module.css` - Demo styles
9. `/src/styles/FAQ.module.css` - FAQ styles
10. `/src/styles/Footer.module.css` - Footer styles

### Updated Files (1 file)
1. `/src/App.tsx` - Added route for new landing page

### Documentation (1 file)
1. `/LANDING_PAGE_SUMMARY.md` - Complete implementation guide

## Component Hierarchy

```
App
└── Router
    ├── Route: "/" → NewLandingPage
    │   ├── Background Effects (blur blobs)
    │   ├── Navigation (sticky)
    │   ├── Hero
    │   │   ├── Headline & Subheadline
    │   │   ├── CTA Buttons
    │   │   ├── Feature Badges
    │   │   └── Product Mock Card
    │   ├── Stats
    │   │   └── Stats Grid (3 cards)
    │   ├── Problem
    │   │   ├── Problem Cards (3)
    │   │   └── Solution Box
    │   ├── Architecture
    │   │   └── Pipeline Steps (3)
    │   ├── Model
    │   │   ├── Architecture Diagram
    │   │   ├── Metrics Grid (4)
    │   │   └── Highlights (2)
    │   ├── Explainability
    │   │   ├── Technique Cards (2)
    │   │   └── Benefits Grid (4)
    │   ├── Demo
    │   │   ├── Tab Navigation (3)
    │   │   └── Tab Content
    │   │       ├── ECG View
    │   │       ├── Risk View
    │   │       └── Explanation View
    │   ├── FAQ
    │   │   └── FAQ Items (6)
    │   └── Footer
    │       ├── Brand & Links
    │       ├── Disclaimer
    │       └── Social Links
    │
    ├── Route: "/old" → LandingPage (original)
    └── Route: "/analysis" → PatientAnalysis
```

## CSS Module Naming Convention

All styles use CSS Modules with BEM-inspired naming:

```css
/* Section container */
.section { }

/* Section elements */
.sectionContainer { }
.sectionLabel { }
.title { }
.subtitle { }

/* Card patterns */
.card { }
.cardHeader { }
.cardBody { }
.cardFooter { }

/* State modifiers */
.active { }
.open { }
.high { }
.medium { }
.low { }
```

## Responsive Breakpoints

```css
/* Desktop-first approach */

@media (max-width: 1024px) {
  /* Tablet: 2-column grids, adjusted layouts */
}

@media (max-width: 768px) {
  /* Mobile: single column, stacked layouts */
}
```

## Color Palette

```css
/* Primary Brand */
--indigo-500: #6366f1;
--indigo-600: #4f46e5;
--purple-500: #8b5cf6;
--purple-600: #7c3aed;

/* Semantic Colors */
--red-500: #ef4444;      /* High risk */
--amber-500: #f59e0b;    /* Medium risk */
--green-500: #10b981;    /* Low risk */

/* Neutrals */
--slate-950: #0f172a;    /* Dark text */
--slate-800: #1e293b;    /* Headings */
--slate-700: #334155;
--slate-600: #475569;    /* Body text */
--slate-500: #64748b;    /* Muted text */
--slate-400: #94a3b8;    /* Labels */
--slate-300: #cbd5e1;
--slate-200: #e2e8f0;    /* Borders */
--slate-100: #f1f5f9;
--slate-50: #f8fafc;

/* Backgrounds */
--bg-primary: #fafbff;   /* Page background */
--bg-secondary: #f5f7ff; /* Section alternates */
```

## Typography Scale

```css
/* Headings */
h1: 4rem (64px)     font-weight: 700
h2: 3rem (48px)     font-weight: 700
h3: 2rem (32px)     font-weight: 700
h4: 1.5rem (24px)   font-weight: 700

/* Body */
Large: 1.25rem (20px)
Base: 1rem (16px)
Small: 0.9375rem (15px)
XSmall: 0.875rem (14px)
Label: 0.75rem (12px)

/* Line Heights */
Tight: 1.1-1.2
Normal: 1.5-1.6
Relaxed: 1.7
```

## Border Radius Scale

```css
Small: 8px
Medium: 12px
Large: 16px
XLarge: 20px
2XLarge: 24px
```

## Spacing Scale (based on 0.25rem = 4px)

```css
0.5rem  = 8px
0.75rem = 12px
1rem    = 16px
1.25rem = 20px
1.5rem  = 24px
2rem    = 32px
3rem    = 48px
4rem    = 64px
6rem    = 96px
8rem    = 128px
```

## Shadow Scale

```css
/* Card shadows */
sm:  0 1px 3px rgba(0, 0, 0, 0.08)
md:  0 4px 12px rgba(0, 0, 0, 0.08)
lg:  0 12px 24px rgba(0, 0, 0, 0.08)
xl:  0 20px 60px rgba(0, 0, 0, 0.08)

/* Brand shadows */
indigo: 0 4px 16px rgba(99, 102, 241, 0.3)
```

---

**Total Lines of Code**: ~3,500+ lines
**Components**: 10 React components
**CSS Modules**: 10 stylesheets
**Implementation Time**: Complete working implementation
**Browser Support**: Modern browsers (ES2020+)

