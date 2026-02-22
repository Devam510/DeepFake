# How It Works

*A plain-language explanation of what this tool does and why uncertainty is unavoidable.*

---

## What We Analyze

When you upload an image, we analyze **statistical patterns** in the image data. These include:

- **Frequency patterns** — How image detail is distributed across different scales
- **Texture characteristics** — Statistical properties of surface patterns
- **Edge consistency** — How sharp transitions appear in the image
- **Noise patterns** — The distribution of subtle variations in the image

These patterns can sometimes differ between AI-generated images and photographs, though not always in detectable ways.

---

## How We Generate a Probability

The analysis produces a single number: the **synthetic probability**.

This represents:
> "Given the patterns we detected, and based on what the model learned from training data, how likely is it that this image exhibits characteristics associated with AI-generated content?"

This is **not** the same as:
> "Is this image AI-generated?"

The difference is important.

---

## Why Uncertainty Exists

### 1. Overlap in Patterns

AI-generated images and photographs can share similar statistical patterns. There is no single feature that definitively identifies AI generation.

```
          Photographs              AI-Generated
              ●●●                     ●●●
            ●●●●●●●                 ●●●●●●●
          ●●●●●●●●●●●             ●●●●●●●●●●●
         ●●●●●●●●●●●●●           ●●●●●●●●●●●●●
              └───────── OVERLAP ─────────┘
```

Many images fall in the overlap zone where patterns are ambiguous.

### 2. Limited Training Data

The model learned from a specific set of examples. It may not recognize:
- New AI generation methods
- Image types not in training
- Edge cases

### 3. Processing Artifacts

Real-world images undergo processing (compression, resizing, filters) that can:
- Add patterns similar to AI artifacts
- Remove patterns the model looks for
- Create misleading signals

### 4. No Ground Truth

We cannot know with certainty which images are truly "authentic." The images labeled as real in training were **assumed** to be non-synthetic, but were not verified.

---

## What "Synthetic Likelihood" Means

When we say an image has "HIGH synthetic likelihood," we mean:

✅ The image exhibits patterns the model associates with AI generation  
✅ Based on training data, similar patterns appeared more often in synthetic images  
✅ The statistical signal is relatively strong  

We do **NOT** mean:

❌ The image is definitely AI-generated  
❌ We have proven its origin  
❌ You should act on this as fact  

---

## What "Synthetic Likelihood" Does NOT Mean

### It Does Not Prove Origin

A high probability does not prove an image was created by AI. It means the image has patterns that, in our training data, were more common in AI-generated images.

### It Does Not Prove Authenticity

A low probability does not prove an image is a genuine photograph. Many AI-generated images may have low synthetic likelihood scores.

### It Is Not a Verdict

This is not a classification. We do not say "this is AI-generated" or "this is real." We only provide a probability with uncertainty.

---

## The Confidence Interval

Along with the probability, we provide a **confidence interval** (e.g., 65%–79%).

This represents:
> "Based on statistical uncertainty, the true probability likely falls somewhere in this range."

A wider interval means more uncertainty. A narrower interval means the model is more confident in its estimate (though still not certain of truth).

---

## The Uncertainty Zone

Probabilities between approximately **30% and 70%** are in the "uncertainty zone."

In this range:
- The model cannot distinguish reliably
- The result is essentially uninformative
- You should treat this as "no meaningful signal"

We explicitly flag results in this zone as **INCONCLUSIVE**.

---

## Why We Cannot Prove Authenticity

**Authenticity requires knowing the origin of an image.**

This tool analyzes:
- Statistical patterns in pixel data

This tool cannot access:
- Camera metadata (which can be faked)
- Chain of custody
- Original source files
- Creator identity
- Intent

Proving authenticity requires forensic investigation, provenance tracking, and human judgment — none of which this tool provides.

---

## Summary

| What We Do | What We Don't Do |
|------------|------------------|
| Analyze statistical patterns | Determine truth |
| Provide probability estimates | Provide verdicts |
| Quantify uncertainty | Prove origin |
| Flag inconclusive results | Make decisions for you |
| Explain limitations | Claim accuracy |

---

*This tool is a starting point for inquiry, not an endpoint for judgment.*
