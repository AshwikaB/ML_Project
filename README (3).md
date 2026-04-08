# Resume Intelligence Platform
> ML-powered resume screening tool with multi-model comparison, skill gap analysis, and an interactive Gradio UI — built for Google Colab.

---

## Overview

The Resume Intelligence Platform is a machine learning project that automates resume-to-job-description matching. It trains and benchmarks four classical ML classifiers on a synthetic + real-data augmented dataset, selects the best-performing model, and exposes the entire pipeline through a clean dark-themed Gradio interface.

Upload a resume (PDF or DOCX), paste a job description, and the platform returns a match score, a Good Fit / Not a Fit verdict, matched and missing skills with color-coded tags, actionable improvement suggestions, and model performance visualizations.

---

## Features

- **Multi-Model Training** — Trains Logistic Regression, Naive Bayes, Linear SVC, and Random Forest in parallel; auto-selects the best model by cross-validated F1 score.
- **Hybrid Scoring** — Combines ML model probability (60%) with TF-IDF cosine similarity (40%), with score correction to prevent inflated results on unrelated pairs.
- **Skill Gap Analysis** — Extracts and maps 100+ tech skills using a canonical synonym taxonomy, returning matched skills, missing skills, and prioritized suggestions.
- **Synthetic Dataset Generator** — Builds a ~2,500 sample balanced dataset with strong matches, hard negatives, and borderline cases — with optional HuggingFace real-data augmentation.
- **Visualization Engine** — Generates F1 bar charts, confusion matrices, and a score gauge chart rendered inline in the UI.
- **Resume Validator** — Rejects non-resume files before scoring using keyword-based structural checks.
- **Gradio UI** — Three-tab dark-themed interface: Input, Results, and Model Insights.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML & NLP | scikit-learn, NLTK, TF-IDF, cosine similarity |
| File Parsing | PyPDF2, python-docx |
| Data | pandas, NumPy, HuggingFace `datasets` |
| Visualization | matplotlib, seaborn |
| UI | Gradio |
| Environment | Google Colab / Jupyter Notebook |

---

## Project Structure

```
resume_intelligence_platform_final.py
│
├── SECTION 0   Installation (Colab pip commands)
├── SECTION 1   Imports
├── SECTION 2   NLTK Downloads
├── SECTION 3   Constants & Skill Knowledge Base (100+ skills + synonym map)
├── SECTION 4   Synthetic Dataset Generator (~2500 samples)
├── SECTION 5   Text Preprocessor (cleaning, lemmatization, stopword removal)
├── SECTION 6   Resume Validator (structural keyword checks)
├── SECTION 7   File Parser (PDF + DOCX extraction)
├── SECTION 8   Skill Extractor (canonical skill mapping)
├── SECTION 9   Model Trainer (4 classifiers, StratifiedKFold CV, auto-select)
├── SECTION 10  Resume Matcher (hybrid scoring pipeline)
├── SECTION 11  Visualization Engine (F1 chart, confusion matrix, gauge)
├── SECTION 12  Gradio UI (3-tab dark interface)
└── SECTION 13  Entrypoint
```

---

## Quick Start (Google Colab)

**Step 1 — Install dependencies**
```bash
!pip install gradio scikit-learn nltk PyPDF2 python-docx matplotlib seaborn datasets -q
```

**Step 2 — Run the script**
```bash
!python resume_intelligence_platform_final.py
```

Or open `Resume_Intelligence_Platform_v4_FINAL.ipynb` and run all cells.

**Step 3 — Use the UI**

Gradio will launch a public link. Open it in your browser, upload a resume, paste a job description, and click **Analyze**.

---

## Scoring Logic

```
Final Score = (0.6 × Model Probability) + (0.4 × Cosine Similarity)
```

**Score Correction Rules:**
- If cosine similarity < 5% → model probability is capped at 35% to prevent inflated scores on unrelated pairs.
- Verdict of **Good Fit** requires Final Score ≥ 55% AND cosine similarity ≥ 5%.

| Verdict | Condition |
|---|---|
| ✅ Good Fit | Score ≥ 55% and cosine ≥ 5% |
| ❌ Not a Fit | Score < 55% or cosine < 5% |

---

## Models Trained

| Model | Notes |
|---|---|
| Logistic Regression | Strong baseline for text classification |
| Multinomial Naive Bayes | Fast, effective on TF-IDF features |
| Linear SVC (Calibrated) | High accuracy on high-dimensional sparse data |
| Random Forest | Ensemble baseline for comparison |

Best model is selected automatically by cross-validated F1 score.

---

## Dataset

The platform generates a synthetic dataset of ~2,500 labeled resume–JD pairs:

- **Strong Matches (label=1)** — Same-role resume and JD pairs across multiple experience levels with noise injection.
- **Hard Negatives (label=0)** — Unrelated role pairs (e.g., ML Engineer resume vs. DevOps JD).
- **Borderline Cases** — Partial-overlap pairs to improve decision boundary learning.
- **Real Data Augmentation** — Optional HuggingFace dataset integration for additional clean matched samples.

---

## Acknowledgements

Built as an ML Viva Project. Inspired by real-world ATS (Applicant Tracking System) tools.
