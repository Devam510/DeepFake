# AI-Generated Image Risk Assessment - UI Design Specification

> **Mode**: SAFE UX DESIGN  
> **Purpose**: Risk assessment interface (NOT marketing)  
> **Core Question**: "What is the estimated risk that this image was AI-generated, given known limitations and uncertainty?"

---

## 1. PAGE LAYOUT

```
┌─────────────────────────────────────────────────────────────────┐
│                         HEADER                                   │
│  "Synthetic Image Risk Estimator"                               │
│  Subtitle: "Statistical analysis with uncertainty"              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     UPLOAD SECTION                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                          │   │
│  │              [ Drop image here or click ]                │   │
│  │                                                          │   │
│  │  Supported: JPEG, PNG, WebP • Max 10 MB • Max 4096px    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ⚠️ Results are probabilistic and may be inconclusive.         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   RISK VISUALIZATION                             │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                                                           │  │
│  │   LOW ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ HIGH     │  │
│  │   0%        30%      50%       70%              100%     │  │
│  │              └─────────────────┘                         │  │
│  │                 UNCERTAINTY                              │  │
│  │                    ZONE                                  │  │
│  │                                                          │  │
│  │           ▲ 72% [65% - 79%]                              │  │
│  │                                                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Confidence Interval: 65% – 79% (95% level)                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  INTERPRETATION PANEL                            │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  MODERATE-HIGH SYNTHETIC LIKELIHOOD                     │    │
│  │                                                         │    │
│  │  This score reflects statistical patterns in the image, │    │
│  │  not proof of origin. The model detected signals        │    │
│  │  commonly associated with AI-generated content.         │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  (If borderline: yellow banner with "INCONCLUSIVE" label)       │
└─────────────────────────────────────────────────────────────────┘

┌────────────────────────────┬────────────────────────────────────┐
│   WHAT THIS MEANS          │   WHAT THIS DOES NOT MEAN          │
├────────────────────────────┼────────────────────────────────────┤
│ ○ Statistical likelihood   │ ✗ Proof of authenticity            │
│ ○ Model confidence level   │ ✗ Proof of AI generation           │
│ ○ Known signal strength    │ ✗ Legal or policy decision         │
│ ○ Patterns in training     │ ✗ Verification of source           │
│   data                     │ ✗ Definitive classification        │
└────────────────────────────┴────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              LIMITATIONS & UNCERTAINTY                           │
│                                                                  │
│  ⚠️ Important Context                                           │
│                                                                  │
│  • Model trained on StyleGAN face images only                   │
│  • Training "real" images were unverified                       │
│  • Performance varies by compression and resolution             │
│  • Not all AI generators leave detectable patterns              │
│                                                                  │
│  ⛔ This tool should NOT be used as sole evidence for           │
│     legal, journalistic, or enforcement decisions.              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         FOOTER                                   │
│  Version 1.0.0 • Model v1.0.0 • For research purposes only      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. COMPONENT HIERARCHY

```
<App>
├── <Header>
│   ├── <Title> "Synthetic Image Risk Estimator"
│   └── <Subtitle> "Statistical analysis with uncertainty"
│
├── <UploadSection>
│   ├── <DropZone>
│   ├── <FormatLimits>
│   └── <ProbabilisticWarning>
│
├── <ResultsContainer> (hidden until analysis complete)
│   │
│   ├── <RiskVisualization>
│   │   ├── <RiskGauge>
│   │   │   ├── <UncertaintyZone> (30-70% highlighted)
│   │   │   ├── <ProbabilityMarker>
│   │   │   └── <ConfidenceBar>
│   │   └── <ConfidenceIntervalText>
│   │
│   ├── <InterpretationPanel>
│   │   ├── <LikelihoodLabel>
│   │   ├── <ExplanationText>
│   │   └── <InconclusiveBanner> (conditional)
│   │
│   ├── <MeaningComparison>
│   │   ├── <WhatThisMeans>
│   │   └── <WhatThisDoesNotMean>
│   │
│   └── <LimitationsPanel>
│       ├── <LimitationsList>
│       └── <EnforcementWarning>
│
├── <ErrorPanel> (conditional)
│   ├── <ErrorIcon>
│   ├── <ErrorTitle>
│   ├── <ErrorMessage>
│   └── <RetryButton>
│
└── <Footer>
    └── <VersionInfo>
```

---

## 3. USER FLOW

```
┌─────────────┐
│   Landing   │
│   (Upload)  │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│  Validating │────▶│   ERROR     │
│    File     │     │   STATE     │
└──────┬──────┘     └─────────────┘
       │
       ▼
┌─────────────┐
│  Uploading  │
│  (Spinner)  │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│  Analyzing  │────▶│   ERROR     │
│  (Progress) │     │   STATE     │
└──────┬──────┘     └─────────────┘
       │
       ▼
┌─────────────────────────────────┐
│                                 │
│   ┌─────────────────────────┐   │
│   │ LOW PROBABILITY (< 30%) │   │
│   │ Blue theme, calm tone   │   │
│   └─────────────────────────┘   │
│                                 │
│   ┌─────────────────────────┐   │
│   │ BORDERLINE (30-70%)     │   │
│   │ Yellow theme, caution   │   │
│   │ "INCONCLUSIVE" banner   │   │
│   └─────────────────────────┘   │
│                                 │
│   ┌─────────────────────────┐   │
│   │ HIGH PROBABILITY (>70%) │   │
│   │ Orange theme, measured  │   │
│   │ Still probabilistic     │   │
│   └─────────────────────────┘   │
│                                 │
└─────────────────────────────────┘
```

---

## 4. EXACT COPY TEXT

### 4.1 HEADINGS

| Location | Text |
|----------|------|
| Page Title | Synthetic Image Risk Estimator |
| Page Subtitle | Statistical analysis with uncertainty |
| Upload Section | Upload an image for analysis |
| Risk Section | Synthetic Probability Estimate |
| Interpretation | Interpretation |
| Meaning Left | What This Means |
| Meaning Right | What This Does NOT Mean |
| Limitations | Limitations & Uncertainty |

---

### 4.2 UPLOAD SECTION COPY

**Drop Zone Text:**
```
Drop image here or click to select
```

**Format Limits:**
```
Supported: JPEG, PNG, WebP • Maximum 10 MB • Maximum 4096 × 4096 pixels
```

**Probabilistic Warning (always visible):**
```
⚠️ Results are probabilistic and may be inconclusive.
   This tool estimates statistical patterns, not origin.
```

---

### 4.3 RISK VISUALIZATION COPY

**Gauge Labels:**
```
LOW                    UNCERTAIN                    HIGH
0%        30%            50%           70%         100%
```

**Confidence Interval:**
```
Confidence Interval: [lower]% – [upper]% (95% level)
```

**Uncertainty Zone Tooltip:**
```
The shaded region (30%–70%) represents outcomes where 
interpretation is unreliable. Results in this range 
should be treated as inconclusive.
```

---

### 4.4 INTERPRETATION PANEL COPY

**Likelihood Labels (verbatim from API):**

| Probability | Label |
|-------------|-------|
| 0% – 20% | LOW synthetic likelihood |
| 20% – 40% | MODERATE-LOW synthetic likelihood |
| 40% – 60% | INCONCLUSIVE – insufficient evidence |
| 60% – 80% | MODERATE-HIGH synthetic likelihood |
| 80% – 100% | HIGH synthetic likelihood |

**Explanation Text (always shown):**
```
This score reflects statistical patterns in the image, 
not definitive proof of origin. The model detected signals 
commonly associated with AI-generated or non-AI content 
based on limited training data.
```

**Borderline Banner (shown when 30% ≤ p ≤ 70%):**
```
⚠️ INCONCLUSIVE — Interpretation Unreliable

The probability falls within the uncertainty zone where 
the model cannot distinguish reliably. This result should 
not be used for any decision-making.
```

---

### 4.5 WHAT THIS MEANS / DOES NOT MEAN

**Left Column (What This Means):**
```
○ Statistical likelihood based on image patterns
○ Model's confidence given its training data
○ Strength of signals the model was trained to detect
○ Comparison to known synthetic image characteristics
```

**Right Column (What This Does NOT Mean):**
```
✗ Proof that the image is or is not AI-generated
✗ Verification of the image source or creator
✗ Legal evidence of authenticity or manipulation
✗ Definitive classification suitable for enforcement
✗ Guarantee of accuracy or correctness
```

---

### 4.6 LIMITATIONS & UNCERTAINTY COPY

**Section Content (always visible):**
```
⚠️ Important Context

• Model trained on StyleGAN face images only — may not 
  generalize to other AI generators or image types.

• Training images labeled "real" were unverified — 
  the model has no ground truth for authentic images.

• Performance varies by image compression, resolution, 
  and post-processing applied after generation.

• Not all AI-generated images leave detectable statistical 
  patterns. Absence of signals does not prove authenticity.

⛔ This tool should NOT be used as sole evidence for 
   legal, journalistic, or enforcement decisions. 
   Always combine with human review and additional 
   verification methods.
```

---

### 4.7 TOOLTIPS

| Element | Tooltip Text |
|---------|--------------|
| Probability Value | "This percentage represents the model's estimate that the image exhibits patterns consistent with AI generation. It is not a certainty score." |
| Confidence Interval | "The true probability likely falls within this range, based on statistical uncertainty in the model's estimate." |
| Uncertainty Zone | "Results in this range (30%–70%) are considered unreliable. The model cannot confidently distinguish between synthetic and non-synthetic patterns." |
| Limitations Icon | "Click to learn about known limitations and appropriate use of this tool." |

---

### 4.8 ERROR MESSAGES

**Unsupported File:**
```
Unsupported File Format

This tool accepts JPEG, PNG, and WebP images only.
Please select a different file.

[Select Different File]
```

**File Too Large:**
```
File Size Exceeds Limit

Maximum file size is 10 MB. 
Please select a smaller image or compress the file.

[Select Different File]
```

**Resolution Too High:**
```
Image Resolution Too High

Maximum supported resolution is 4096 × 4096 pixels.
Please resize the image before uploading.

[Select Different File]
```

**API Error:**
```
Analysis Could Not Complete

The service encountered an issue processing your request.
This may be temporary. Please try again in a moment.

Request ID: [request_id]

[Try Again]
```

**Timeout:**
```
Analysis Timed Out

The request took longer than expected to complete.
This may be due to high demand. Please try again.

[Try Again]
```

**Low-Confidence Output (informational, not error):**
```
ℹ️ Low Signal Strength

The model detected weak or ambiguous patterns in this image.
The result should be interpreted with additional caution.
```

---

## 5. STATE LOGIC

### 5.1 UI States by Probability

| Probability Range | Theme Color | Banner | Behavior |
|-------------------|-------------|--------|----------|
| 0% – 30% | Blue (#2563EB) | None | Calm, informative |
| 30% – 70% | Amber (#D97706) | "INCONCLUSIVE" | Caution emphasized |
| 70% – 100% | Orange (#EA580C) | None | Measured, still uncertain |

**Note:** NO GREEN OR RED colors used to avoid binary perception.

---

### 5.2 Component Visibility by State

| Component | Upload | Loading | Low (0-30%) | Borderline (30-70%) | High (70-100%) | Error |
|-----------|--------|---------|-------------|---------------------|----------------|-------|
| Upload Section | ✅ | ❌ | ✅ (collapsed) | ✅ (collapsed) | ✅ (collapsed) | ✅ |
| Loading Spinner | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Risk Visualization | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |
| Interpretation | ❌ | ❌ | ✅ | ✅ + BANNER | ✅ | ❌ |
| Meaning Comparison | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |
| Limitations | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ |
| Error Panel | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

---

### 5.3 Borderline Behavior (CRITICAL)

When probability is between 30% and 70%:

1. **Show prominent banner:**
   - Amber background
   - "INCONCLUSIVE — Interpretation Unreliable"
   
2. **Highlight uncertainty zone on gauge:**
   - Pulse animation on uncertainty region
   - Marker clearly within shaded zone

3. **Add inline warning:**
   - Below interpretation: "This result should not be used for decision-making."

4. **Expand limitations by default:**
   - Limitations section fully visible, not collapsed

---

## 6. VISUAL DESIGN NOTES

### 6.1 Color Palette (Uncertainty-Aware)

| Purpose | Color | Hex |
|---------|-------|-----|
| Low Probability | Slate Blue | #2563EB |
| Borderline | Amber | #D97706 |
| High Probability | Deep Orange | #EA580C |
| Uncertainty Zone | Light Gray | #E5E7EB |
| Background | Off-White | #FAFAFA |
| Text | Dark Gray | #1F2937 |
| Warning | Amber | #F59E0B |

**FORBIDDEN:** Green (#22C55E), Red (#EF4444) — avoid binary perception.

---

### 6.2 Typography

| Element | Font | Size | Weight |
|---------|------|------|--------|
| Page Title | Inter | 28px | 600 |
| Section Headings | Inter | 20px | 600 |
| Body Text | Inter | 16px | 400 |
| Probability Value | Inter | 48px | 700 |
| Labels | Inter | 14px | 500 |
| Warnings | Inter | 14px | 500 |

---

### 6.3 Gauge Design

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   ┌────────┬────────────────────────────┬────────────────┐  │
│   │  BLUE  │     GRAY (UNCERTAINTY)     │    ORANGE     │  │
│   │  0-30  │         30 - 70            │    70-100     │  │
│   └────────┴────────────────────────────┴────────────────┘  │
│                           ▲                                  │
│                          72%                                 │
│                      ┌────────┐                              │
│                      │ marker │                              │
│                      └────────┘                              │
│                                                              │
│   Confidence: [65% ─────────●───────── 79%]                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

- **No gradient from green to red**
- **Uncertainty zone (30-70%) is visually distinct (hatched or dotted)**
- **Confidence interval shown as horizontal error bar**

---

## 7. ACCESSIBILITY

| Requirement | Implementation |
|-------------|----------------|
| Color Blindness | No red/green; use patterns + labels |
| Screen Readers | All images have alt text; ARIA labels on interactive elements |
| Keyboard Nav | Full keyboard navigation for upload and retry |
| Text Contrast | Minimum 4.5:1 ratio (WCAG AA) |
| Focus Indicators | Visible focus rings on all interactive elements |

---

## 8. RESPONSIVE BEHAVIOR

| Breakpoint | Layout |
|------------|--------|
| Desktop (>1024px) | Two-column for Meaning Comparison |
| Tablet (768-1024px) | Stacked layout, gauge full-width |
| Mobile (<768px) | Single column, collapsible sections |

---

**⛔ STOPPED** — Awaiting next phase prompt.  
No legal framing, no monitoring, no deployment.
