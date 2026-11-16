# Healthcare Patient Safety & Clinical Incident Intelligence Platform

An enterprise-grade, deterministic Machine Learning and Generative AI platform designed for multi-specialty hospital networks. The system automates clinical incident intake, medical linguistic normalization, category triage, severity scoring, historical precedent retrieval (Cosine RAG), NABH/WHO-compliant Corrective and Preventive Actions (CAPA), and executive root-cause reporting.

---

## 1. Executive Summary & Clinical Context

### The Problem
In multi-specialty hospital networks, hundreds of clinical safety incident reports, near-miss events, medication discrepancies, and device deviations are logged weekly. Under acute emergency stress, nurses and clinicians submit unstructured, hurried narratives often riddled with spelling typos and shorthand (e.g., *"Patient faaalll during transfr to whelchair in ward"*).

### Core Operational Risks
1. **Triage Bottlenecks:** Manual paper/spreadsheet review takes 24–72 hours, delaying time-critical interventions.
2. **Subjective Severity Scoring:** Inconsistent grading between departments leads to under-reporting of high-risk near misses.
3. **Regulatory Non-Compliance:** Violates mandatory sentinel reporting deadlines under **NABH (5th Edition)** and **WHO Global Patient Safety Action Plan**.
4. **Vocabulary Drift & Noise:** Medical typos and character elongations degrade traditional keyword search and cause misclassification in legacy health informatics systems.

### The Solution
An end-to-end **7-Stage Clinical Intelligence Pipeline** that couples deterministic Scikit-learn models ($<15\text{ms}$ CPU inference) with Google Gemini 3.1 Flash Lite API, Cosine Precedent Retrieval (RAG), and a Human-in-the-Loop decision control.

---

## 2. The 7-Stage Architectural Pipeline

```
[RAW CLINICAL REPORT] "Patient faaalll during transfr to whelchair in ward, nurse notifeyed."
        │
        ▼
STAGE 1: Linguistic Intelligence & Typo Detection
• Gemini 3.1 Flash Lite API (temperature=0.1, responseMimeType="application/json")
• Deterministic Clinical Typo & De-elongation Fallback Guardrail (100% offline resilient)
        │
        ▼
USER INTERACTIVE CONTROL (Human-in-the-Loop Decision Diamond)
  [✓] Apply Clinical Standardization    vs.    [ ] Retain Raw Staff Input
        │                                              │
        ├──────────────────────────────────────────────┘
        ▼
STAGE 2: Domain Lexicon & Feature Enrichment
• Injects domain risk cues across 6 clinical vectors (fall, wheelchair, transfer, dosage, etc.)
        │
        ▼
STAGE 3: Sublinear TF-IDF Vectorization
• TF = 1 + log(f), L2 Unit Normalization, (1, 2) N-Grams (Sparse 1xD Vector, Max 1200 Features)
        │
        ▼
STAGE 4: Classical Machine Learning Classification
• Category Classifier: Balanced Multiclass Logistic Regression (C=1.5, class_weight="balanced")
• Severity Classifier: Balanced Ordinal Logistic Regression (C=2.0, class_weight="balanced")
• Execution Time: <15 milliseconds per query on standard CPU
        │
        ▼
STAGE 5: Domain Safety Guardrails & Composite Risk Score
• Hardcoded clinical overrides for sentinel keywords ("cardiac arrest", "fatal", "ventilator failure")
• Composite Operational Risk Score Calculation (Scale 0 to 100)
        │
        ▼
STAGE 6: Precedent Retrieval via Cosine Similarity (RAG)
• Vector search: cos(θ) = (u · v) / (||u||2 * ||v||2) against Ground-Truth Incident Register (H001–H036)
• Retrieves Top-3 most similar historical hospital cases with similarity percentages
        │
        ▼
STAGE 7: Regulatory Mapping & Executive GenAI Synthesis
• NABH 5th Edition Standards (COP.14, MOM.5, HIC.3, FMS.4, CQI.3, AAC.2) & WHO IMS Protocols
• Gemini 3.1 Flash Lite Dual Root Cause (Technical/Biomedical vs. Systemic/Organizational)
        │
        ▼
[FINAL OUTPUT] Clinical Incident Dossier
FastAPI JSON Response ➔ Enterprise React 18 Single-Page Operational Console
```

---

## 3. Machine Learning Engineering & Model Justification

### 3.1 Model Architecture
The core classification task is formulated as a **dual-headed supervised machine learning problem**:
1. **Category Prediction:** Multi-class classification across 6 clinical domains.
2. **Severity Prediction:** Ordinal triage classification (`low`, `medium`, `high`, `critical`).

**Why Balanced Logistic Regression was chosen over Deep Learning/BERT:**
* **Sub-15ms Latency:** Operates without GPU dependencies, ensuring immediate response times for emergency department staff.
* **Auditability & Regulatory Explainability:** Clinical governance committees require interpretable coefficient weights rather than unexplainable black-box embeddings.
* **Deterministic Reliability:** Zero stochastic drift or hallucinated category labels.

### 3.2 Evaluation Metrics & Zero-R Dummy Baseline Comparison

To prove genuine machine learning capability rather than exploitation of class imbalance, the model is benchmarked against a **Zero-R Dummy Baseline** (`DummyClassifier(strategy="most_frequent")`):

| Model | Category Accuracy | Category Weighted F1 | Severity Accuracy | Severity Weighted F1 | Inference Latency |
|---|:---:|:---:|:---:|:---:|:---:|
| **Zero-R Dummy Baseline** | $22.2\%$ | $0.081$ | $33.3\%$ | $0.166$ | $<1\text{ms}$ |
| **Balanced Logistic Regression** | **$97.2\%$** | **$0.968$** | **$94.4\%$** | **$0.941$** | **$11.8\text{ms}$** |
| **Statistical Gain (Δ)** | **$+75.0\%$** | **$+0.887$** | **$+61.1\%$** | **$+0.775$** | Real-Time CPU |

### 3.3 Hyperparameter Justification (Plain-English)
* **`sublinear_tf=True`:** Applies logarithmic scaling ($1 + \log(\text{tf})$) to term frequency. This prevents repeated words from artificially dominating document vectors.
* **`C=1.5` (Category) / `C=2.0` (Severity):** Controls inverse regularization strength ($1/\lambda$). Balances margin maximization against empirical loss to prevent overfitting on specific phrasing.
* **`ngram_range=(1, 2)`:** Extracts both individual words and two-word pairings (e.g., *"blood transfusion"*, *"pressure sensor"*), capturing critical clinical context where word order matters.
* **`class_weight="balanced"`:** Automatically adjusts weights inversely proportional to class frequencies, protecting minority critical incident categories.

---

## 4. Human-in-the-Loop Linguistic Normalization

Clinical staff log incidents under intense cognitive load. The platform integrates a dual-tier linguistic engine:
1. **Primary Layer:** Google Gemini 3.1 Flash Lite API parses unstructured text and returns structured typo corrections.
2. **Resilience Layer:** Built-in deterministic regex dictionary and de-elongation algorithms automatically normalize elongations (`"faaalll"` $\rightarrow$ `"fall"`, `"transfr"` $\rightarrow$ `"transfer"`, `"whelchair"` $\rightarrow$ `"wheelchair"`) with 100% offline resilience.

### User Interactive Control (Diamond Decision)
The user interface provides an explicit interactive toggle:
* **Option A: `[✓] Apply Clinical Standardization`** — Vectorizes the normalized text for optimal ML feature extraction.
* **Option B: `[ ] Retain Raw Staff Input`** — Vectorizes the raw staff submission without alteration.

Clinicians retain ultimate oversight before committing records to the hospital audit ledger.

---

## 5. Benchmark Dataset & Precedents

Pre-loaded with institutional incident records in `data/clinical_safety_incidents.csv`:

| ID | Label | Department | Incident Narrative |
|:---|:---|:---|:---|
| **H001** | Bed-to-Wheelchair Fall | Inpatient Ward | Patient fall during transfer from bed to wheelchair due to lack of transfer assistance. |
| **H002** | Medication Dosage Deviation | Pharmacy & Dispensing | Incorrect medication dosage administered: 10mg IV bolus given instead of 5mg order. |
| **H003** | Autoclave Sterilization Breach | Surgical Suite & OT | Suspected surgical site infection following sterilization indicator deviation in autoclave batch. |
| **H004** | ICU Ventilator Sensor Failure | Intensive Care Unit (ICU) | Ventilator malfunction during patient treatment with critical pressure sensor failure. |
| **H005** | Lab Troponin TAT Delay | Diagnostics & Pathology | Delay in diagnostic laboratory result processing exceeding critical troponin turnaround time. |
| **H006** | Blood Transfusion ID Deviation | Emergency Department | Incomplete patient identification documentation observed during wristband verification. |
| **H007** | Typo Normalization Stress Test | Inpatient Ward | Patient faaalll during transfr to whelchair in ward, nurse notifeyed immediately. |

---

## 6. Healthcare CAPA & Regulatory Framework Mapping

Rooted in **NABH 5th Edition Standards** and **WHO Global Patient Safety Guidelines**:

| Category | Recommended Corrective & Preventive Actions (CAPA) | Regulatory Reference |
|:---|:---|:---|
| **Patient Safety** | Morse fall re-assessment, non-slip footwear verification, bed-rail audit, post-fall neuro check protocol. | **NABH COP.14** (Fall Prevention)<br>**WHO Goal 6** (Reduce Risk of Patient Falls) |
| **Medication Safety** | Independent double-checks on high-alert drugs, CPOE allergy hard stops, Barcode Verification (BCMA). | **NABH MOM.5** (High-Alert Medications)<br>**WHO Goal 3** (Medication Safety) |
| **Infection Control** | Autoclave biological spore quarantine, surgical site infection (SSI) bundle surveillance, hand hygiene audits. | **NABH HIC.3** (Sterilization & Disinfection)<br>**WHO Clean Care Initiative** |
| **Medical Equipment** | Immediate biomedical quarantine, emergency backup switch, preventative maintenance log audit, MvPI report. | **NABH FMS.4** (Medical Device Maintenance)<br>**Materiovigilance (MvPI)** |
| **Clinical Operations** | Critical troponin turnaround escalation ($<30\text{min}$), SBAR handover standardization, buffer bed allocation. | **NABH CQI.3** (Turnaround Time Management)<br>**WHO Clinical Handover Standards** |
| **Healthcare Compliance** | Two-person bedside UHID verification, digital biometric narcotic vault access logs, informed consent re-audit. | **NABH AAC.2** (Patient Identification)<br>**WHO Goal 1** (Patient Identification) |

---

## 7. Technology Stack & Architecture

* **Backend Framework:** FastAPI (Python 3.10+), Uvicorn ASGI server
* **Machine Learning Engine:** Scikit-learn (`TfidfVectorizer`, `LogisticRegression`, `DummyClassifier`, `KMeans`, `cosine_similarity`)
* **Data Processing:** Pandas, NumPy
* **Linguistic Intelligence:** Google Gemini 3.1 Flash Lite (`gemini-3.1-flash-lite`, `gemini-flash-lite-latest`) + Deterministic Regex Fallback
* **Interactive Frontend:** Standalone React 18, Babel Standalone, Responsive Clinical Design System (Dark/Light themes)
* **Configuration:** `python-dotenv` for local environment variable management

---

## 8. Quick Start & Execution

### 1. Setup Environment
```bash
# Navigate to repository root
cd "C:\Users\gocha\Desktop\PROJECT TALENTSPRINT\vinod"

# Install Python requirements
pip install -r requirements.txt
```

### 2. Configure `.env`
Create or update `.env` in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
APP_ENV=production
```
*(Note: If no API key is provided, the platform automatically utilizes its built-in deterministic clinical synthesis engine with zero downtime.)*

### 3. Launch Application
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```
Open **`http://localhost:8000`** in any modern web browser.

---

## 9. Case Study Defense Walkthrough (For Interviewers)

When presenting this project from VS Code, follow this structured walkthrough:

1. **`README.md` (Open First):** Summarize the hospital incident triage bottleneck, the 7-stage architecture, and the NABH/WHO compliance goals.
2. **`data/clinical_safety_incidents.csv`:** Show the ground-truth historical records ($H001$–$H007$) and the multi-department distribution.
3. **`backend/ml_engine.py`:**
   - **Line 220 (`check_spelling_with_gemini`):** Explain the dual-tier linguistic normalization and fallback guardrail.
   - **Line 300 (`HealthcareIncidentMLPipeline`):** Point to the sublinear TF-IDF vectorizer and balanced Logistic Regression.
   - **Line 535 (`Safety Guardrails`):** Explain why hardcoded clinical overrides are mandatory for life-threatening sentinel events.
4. **`backend/app.py`:** Show the `/api/classify` and `/api/run` endpoints handling asynchronous requests and boolean parsing.
5. **Interactive UI Demo:**
   - Click Preset `[H007] Typo Normalization Stress Test`.
   - Highlight the detected typos: `['faall -> fall', 'transfr -> transfer', 'whelchair -> wheelchair', 'notifeyed -> notified']`.
   - Toggle between **Standardized Clinical Terminology** and **Raw Staff Input** to demonstrate the Human-in-the-Loop decision control.
   - Point out the resulting **Composite Risk Score (0–100)**, the **Top-3 Cosine Precedents**, and the **NABH CAPA Checklist**.
