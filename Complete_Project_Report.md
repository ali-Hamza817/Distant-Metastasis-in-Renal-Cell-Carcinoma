# Distant Metastasis Prediction in Renal Cell Carcinoma (RCC)
## Complete Project Report — Honest, No Embellishments

> **Date:** June 2026
> **Author:** Ali Hamza
> **Repository:** [ali-Hamza817/Distant-Metastasis-in-Renal-Cell-Carcinoma](https://github.com/ali-Hamza817/Distant-Metastasis-in-Renal-Cell-Carcinoma)
> **Status:** Prototype / Research Stage

---

## Table of Contents

1. [Project Purpose & Motivation](#1-project-purpose--motivation)
2. [Datasets — What They Are & How They Were Obtained](#2-datasets)
3. [The Patient ID Linkage Problem — The Core Honest Limitation](#3-dataset-linkage-problem)
4. [Exploratory Data Analysis (EDA)](#4-eda)
5. [Architecture Design](#5-architecture)
6. [Training — What Actually Happened](#6-training)
7. [Actual Results — No Lies](#7-results)
8. [The Streamlit Application — What It Actually Does](#8-streamlit)
9. [What Is Done vs. What Is Remaining](#9-done-vs-remaining)
10. [Main Contributions](#10-contributions)
11. [Future Steps](#11-future-steps)
12. [Project File Structure](#12-file-structure)

---

## 1. Project Purpose & Motivation

**Renal Cell Carcinoma (RCC)** — specifically Clear Cell RCC — is the most common type of kidney cancer. When it spreads (metastasizes) to distant organs like the lungs, liver, brain, or bones, the 5-year survival rate drops from ~75% to ~12%.

**The clinical problem:** Clinicians currently use staging (T/N/M classification), histology, and experience to predict whether a tumor will metastasize. This is unreliable. There is no validated AI tool that **fuses CT imaging, RNA expression profiles, and clinical data** to predict distant metastasis probability at diagnosis.

**The research objective of this project:**

> Build a multimodal deep learning framework that takes a patient's CT scan, RNA-seq genomic profile, and standard clinical data, and outputs a probability of distant metastasis, a survival risk score, and a clinical routing recommendation.

**Why multimodal?** Each data type captures a different dimension of tumor biology:

| Modality | What It Captures |
|---|---|
| CT Imaging | Morphology, invasion patterns, vascular features |
| RNA-seq / Genomics | Molecular aggressiveness, gene expression patterns |
| Clinical Data | Age, stage, grade, tumor size — proven survival predictors |

The hypothesis is that fusing all three gives more predictive power than any single modality alone.

---

## 2. Datasets — What They Are & How They Were Obtained

Three separate, publicly available datasets were used. **They do not share patient IDs — this is the central limitation documented in Section 3.**

---

### 2.1 SEER — The Clinical Dataset

| Property | Detail |
|---|---|
| **Full Name** | Surveillance, Epidemiology, and End Results Program |
| **Maintained by** | National Cancer Institute (NCI), USA |
| **What it contains** | Age, sex, race, tumor stage (T/N/M), histology type, Fuhrman grade, tumor size, survival months, metastasis status at diagnosis |
| **How to access** | Must request access at [seer.cancer.gov](https://seer.cancer.gov). Requires a signed Data Use Agreement (DUA). |
| **How we downloaded it** | After DUA approval, downloaded as a `.txt` fixed-width export file from SEER*Stat software. Converted using a `.dic` data dictionary file to a structured CSV. |
| **What we ended up with** | `seer_rcc_2010_2018_clean.csv` — **36,000+ patient records** for RCC diagnosed 2010–2018. |
| **Key Label** | `metastasis` column — binary (0 = no, 1 = yes). Approximately **6% positive rate.** Highly imbalanced. |

**Raw files in repo:**
- `export.txt` — raw SEER fixed-width export (~29 MB)
- `export.dic` — SEER data dictionary
- `seer_rcc_2010_2018_clean.csv` — cleaned & processed

---

### 2.2 TCIA / TCGA-KIRC — CT Imaging Dataset

| Property | Detail |
|---|---|
| **Full Name** | The Cancer Imaging Archive — TCGA Kidney Renal Clear Cell Carcinoma |
| **Maintained by** | National Cancer Institute / TCIA |
| **What it contains** | DICOM-format 3D CT scans of kidney tumors from ~267 unique patients |
| **How to access** | Freely available at [cancerimagingarchive.net](https://www.cancerimagingarchive.net). Download via NBIA Data Retriever or tcia_utils Python library. |
| **How we downloaded it** | Using the TCIA REST API and `tcia_utils`. Script: `scripts/01_download_tcia.py`. |
| **What we ended up with** | `TCIA_TCGA-KIRC_09-16-2015 (2)/tcga_kirc/` — ~267 patient DICOM folders |
| **Patient IDs** | TCGA-style IDs (e.g., `TCGA-B0-4699`). **Not the same as SEER IDs.** |

---

### 2.3 CPTAC-CCRCC — Genomics Dataset

| Property | Detail |
|---|---|
| **Full Name** | Clinical Proteomic Tumor Analysis Consortium — Clear Cell Renal Cell Carcinoma |
| **Maintained by** | NCI / NIH |
| **What it contains** | RNA-seq gene expression profiles, protein abundance, clinical annotations for ~110 patients |
| **How to access** | Available via Genomic Data Commons (GDC) at [gdc.cancer.gov](https://gdc.cancer.gov). Requires GDC Data Transfer Tool. |
| **How we downloaded it** | Using GDC API. Script: `scripts/02_download_gdc.py`. |
| **What we ended up with** | `cptac_ccrcc/` — folders containing `.tsv` files per patient with gene expression values |
| **Patient IDs** | CPTAC-style IDs (e.g., `C3L-00004`). **Different from both SEER and TCIA.** |

---

## 3. Dataset Linkage Problem — The Core Honest Limitation

**This is the most critical limitation of the entire project. You must understand this fully.**

> The three datasets — SEER, TCIA/TCGA-KIRC, and CPTAC-CCRCC — **do not share a common patient identifier**. They are three completely independent cohorts of different patients.

| Dataset | Patient ID Style | # Patients | Cross-dataset Overlap |
|---|---|---|---|
| SEER | Anonymous registry IDs | 36,000+ | ❌ None with TCIA or CPTAC |
| TCIA / TCGA-KIRC | `TCGA-XX-XXXX` | ~267 | ❌ None with SEER |
| CPTAC-CCRCC | `C3L-XXXXX` | ~110 | Partial with TCGA via biospecimen crosswalk |

**TCGA ↔ CPTAC linkage exists** through the GDC's biospecimen crosswalk files — some CPTAC patients were originally TCGA patients. However, mapping those ~80–100 patients to SEER clinical records requires a **multi-year research-grade data harmonization study**.

**What this means in practice:**
The current system **cannot** take Patient A's SEER record + Patient A's CT scan + Patient A's RNA-seq and run them through the model as a true linked triplet. **Each modality comes from a completely different cohort of patients.**

This is not a bug we can fix in a day. It is a fundamental medical informatics challenge. The correct solution involves:
1. Obtaining TCGA-KIRC patients who exist in both TCIA (imaging) and GDC (genomics)
2. Applying for TCGA clinical data from GDC to get their staging/metastasis labels
3. Building a cross-walk mapping file that links all three

This leaves us with ~80–100 truly linked patients — a very small number for deep learning, but scientifically valid.

---

## 4. Exploratory Data Analysis (EDA)

Performed on the SEER dataset (`seer_rcc_2010_2018_clean.csv`):

| Finding | Detail |
|---|---|
| Total records | ~36,000 patients |
| After dropping NaN on 7 core features | **~267 records** (very strict filter — this became the training set) |
| Metastasis rate (class imbalance) | **~6% positive** (metastasis = 1) |
| Age distribution | Mean ~62 years, range 18–95 |
| Sex split | ~60% Male, ~40% Female |
| Most common T-stage | T3 (locally advanced) |
| Most common Histology | Clear Cell RCC (~75%) |
| Survival months range | 0–107 months |

**Class imbalance is the most critical data challenge.** With only 6% positive labels, a naive model that always predicts "no metastasis" achieves ~94% accuracy but is medically worthless. This drove the decision to use `BCEWithLogitsLoss` with a `pos_weight` correction.

---

## 5. Architecture Design

The full architecture is defined in `models/swin3d_fusion.py`. There are **two versions**: the full intended architecture (coded but not trained on real data) and the proxy version that was actually trained.

---

### 5a. What a Real 3D Swin Transformer Is

The **Swin Transformer** (Liu et al., 2021, Microsoft Research) is a hierarchical Vision Transformer that uses **Shifted Window attention** instead of global self-attention.

In its 3D form, it processes volumetric data like CT scans:
- **Input:** `(Batch, 1, Depth, Height, Width)` — a 3D CT volume
- **Step 1:** Divide the volume into non-overlapping 3D patches (e.g., 4×4×4 voxels) → Patch Embedding
- **Step 2:** Window-based Multi-Head Self-Attention within local 3D windows
- **Step 3:** Shifted windows in alternate layers for cross-window communication
- **Step 4:** 3–4 hierarchical stages, each doubling channel depth via Patch Merging
- **Output:** 768-dimensional feature vector per patient

A real 3D Swin Transformer requires:
- Pre-trained weights (trained on large video or CT datasets)
- Significant GPU RAM (typically >24 GB for full-resolution CT)
- Full DICOM preprocessing: resampling, registration, tumor cropping

**The full implementation is coded in `models/swin3d_fusion.py`** and includes:
- `PatchEmbed3D` — volumetric patch embedding via 3D Conv
- `WindowAttention3D` — multi-head self-attention with relative position bias
- `SwinBlock3D` — pre-norm residual blocks with attention + MLP
- `PatchMerging3D` — spatial downsampling with channel doubling
- 3 stages with configurable depth (2,2,2) and heads (3,6,12)

---

### 5b. What Was Actually Trained — The Proxy 3D CNN

Because we only had 267 CT patients, no aligned data, and an RTX A2000 12GB, the training script (`scripts/final_gpu_train.py`) uses a **Proxy 3D CNN** that produces the same tensor shapes as the Swin but without shifted-window attention:

```python
class SwinTransformer3D(nn.Module):
    # Proxy — NOT a real Swin Transformer
    def __init__(self, out_dim=768):
        super().__init__()
        self.conv = nn.Conv3d(1, 16, kernel_size=7, stride=4, padding=3)
        self.swin_blocks = nn.Sequential(
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.fc = nn.Linear(32, out_dim)
```

**This is a 3-layer 3D CNN, not a Swin Transformer.** It is called `SwinTransformer3D` for naming consistency only.

---

### 5c. The Genomics Transformer

Implemented in both `models/swin3d_fusion.py` and `scripts/final_gpu_train.py`. This is mathematically sound.

**Architecture:**
1. Input: 500-dimensional RNA-seq feature vector `(Batch, 500)`
2. Each gene treated as a token: `Linear(1 → 256)`
3. Learnable positional encodings: `Parameter(torch.randn(1, 500, 256))`
4. 4-layer `TransformerEncoderLayer` (256 dim, 8 heads, 4× feedforward)
5. Attention-weighted pooling over gene tokens
6. Output: `Linear(256 → 768)` + GELU + LayerNorm

**Limitation in current training:** Input to this branch is `torch.zeros(500)` — not real RNA-seq values — because CPTAC data is not linked to SEER labels.

---

### 5d. The Cross-Attention Fusion (Core Scientific Contribution)

Implemented in `models/swin3d_fusion.py` as `CrossAttentionFusion`.

**How it works:**
1. Three feature vectors arrive: imaging (768-dim), genomics (768-dim), clinical (50-dim)
2. All projected to a common 512-dimensional space
3. **Bidirectional cross-attention:**
   - Imaging attends to Genomics
   - Genomics attends to Imaging
   - Clinical attends to both Imaging and Genomics simultaneously
4. Skip connections + LayerNorm after each attention
5. Concatenation: `(Batch, 1536)` → Fusion MLP → `(Batch, 512)`

**Output heads (from 256-dim Shared Dense Layer):**

| Head | Output | Task |
|---|---|---|
| `met_head` | `Linear(256, 1)` | Metastasis probability (sigmoid) |
| `surv_head` | `Linear(256, 1)` | Survival risk score (continuous) |
| `site_head` | `Linear(256, 4)` | Lung / bone / liver / brain spread |
| `decision_head` | `Linear(256, 3)` | Clinical routing (3 classes) |

**Parameter Count:**

| Component | Parameters |
|---|---|
| Proxy 3D CNN (used in training) | ~60,000 |
| Real 3D Swin (coded, not trained on real data) | ~28 million |
| Genomics Transformer | ~1.3 million |
| Clinical Encoder (MLP) | ~45,000 |
| CrossAttention Fusion | ~3.1 million |
| Shared Dense + All Heads | ~660,000 |
| **Total as trained** | **~5.2 million** |
| **Total with real Swin** | **~33 million** |

---

## 6. Training — What Actually Happened

### Hardware
- **GPU:** NVIDIA RTX A2000 12GB
- **Framework:** PyTorch + CUDA
- **Training script:** `scripts/final_gpu_train.py`

### What Data Was Actually Fed Into the Model

| Modality | What Was Claimed | What Was Actually Fed |
|---|---|---|
| Clinical | Real SEER data | ✅ **REAL** — 7 features standardized from `seer_rcc_2010_2018_clean.csv` |
| CT Imaging | Real TCIA DICOMs | ❌ **ZERO TENSORS** — `torch.zeros(1, 32, 32, 32)`. Folders are located but no DICOM-to-volume conversion was performed. |
| Genomics | Real CPTAC RNA-seq | ❌ **ZERO TENSORS** — `torch.zeros(500)`. CPTAC folders are located but values are not loaded. |

**Why zeros instead of random noise?**
An earlier attempt used `torch.randn()` (Gaussian noise). This was worse because:
- Random noise has high gradient variance that overwhelms the real clinical signal
- Zero tensors produce neutral (zero) gradient contributions from the imaging/genomics branches
- This allows the clinical data to drive learning without being actively corrupted

### Dataset Sizes

| Split | Samples |
|---|---|
| Total after NaN removal | 267 patients |
| Training (80%, stratified) | ~213 patients |
| Validation (20%, stratified) | ~54 patients |
| Positive cases (metastasis=1) in training | ~13 patients |
| Negative cases in training | ~200 patients |

### Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 8 |
| **Epochs Trained** | **30** |
| Loss (Metastasis) | `BCEWithLogitsLoss(pos_weight ≈ 15.7)` |
| Loss (Survival) | `MSELoss` |
| Loss (Clinical Decision) | `CrossEntropyLoss` (random placeholder labels) |
| Combined Loss | `L_met + 0.5*L_surv + 0.5*L_clin` |
| Dropout | 0.3 |

### Positive Weight Rationale
With ~16 positive cases out of 267 total:
`pos_weight = (267 - 16) / 16 ≈ 15.7`

This tells the loss function to penalize missing a metastasis case 15.7× more than a false alarm — essential for the model not to collapse to always predicting "no metastasis."

### What Happened at Different Epoch Ranges

| Epoch Range | Observed Behaviour |
|---|---|
| 1–5 | Loss decreasing rapidly, model learning clinical correlations |
| 5–15 | Steady improvement, AUC rising above random (0.50) |
| 15–30 | **Optimal training zone — genuine plateau at ~64% AUC** |
| 30+ | **Mode collapse risk** — model starts predicting all-positive, AUC degrades back toward 0.50 |

**30 epochs was the empirically determined optimal stopping point.**

---

## 7. Actual Results — No Lies

### THE ONLY REAL METRICS

> These were computed on the validation set (54 samples) after 30 epochs of training.
> They are mathematically real — not fabricated.

| Metric | Value | Honest Interpretation |
|---|---|---|
| **ROC AUC** | **0.641 (64.1%)** | Genuine. The model distinguishes metastasis cases 64% of the time vs. 50% random chance. |
| **Accuracy** | **~90.7%** | Technically real BUT misleading — see explanation below |
| Training Loss (final epoch) | Low / converged | Model learned to fit training data |

---

### Why the 90.7% Accuracy Is Misleading

With 54 validation samples at 6% positive rate: ~51 negatives, ~3 positives.

If the model learns to say "no metastasis" for nearly every case (which is the easiest pattern), it immediately achieves:
`51/54 = 94.4% accuracy`

**High accuracy in class-imbalanced medical data is meaningless.** A model that misses every cancer patient is a dangerous model despite showing 94% accuracy.

**ROC AUC is the metric that matters** — it evaluates prediction ranking across all probability thresholds regardless of class balance.

---

### The Previously Fabricated Number

At one point during the project, the terminal output was hardcoded to print `91.4%` accuracy. **That specific number was fabricated and injected into a print statement — not computed.** It has been removed. The 64.1% AUC is the only legitimately computed metric from this project.

---

### Why AUC Is 64% and Not Higher

1. **Only 267 training samples** — far too few for a 5M parameter model
2. **CT and genomics inputs are zero tensors** — only 7 clinical features are actually informing the model
3. **~16 positive examples in training** — model can barely learn the minority pattern
4. **No pre-training** — trained from random weight initialization on 267 samples
5. **The imaging and genomics branches contribute zero gradient** — they are learning nothing

**A LightGBM model trained on all 36,000 SEER records with proper feature engineering would very likely achieve AUC >0.80 — without any deep learning at all.** This remains as the most important near-term step.

---

## 8. The Streamlit Application — What It Actually Does

**File:** `streamlit_app.py`
**Deployed URL:** `distant-metastasis-in-renal-cell-carcinoma-c87sfhzj4hhtjm5rsnk.streamlit.app`

### What the App Does When You Click "Run"

It executes this formula:

```python
base_risk = (t_stage * 0.15) + (n_stage * 0.2) + (grade * 0.1)
met_prob = min(max(base_risk + np.random.uniform(-0.1, 0.1), 0.05), 0.95)
survival_score = max(100 - (met_prob * 80) + np.random.uniform(-5, 5), 10)
```

**This is a hand-crafted heuristic formula. It is NOT a neural network inference.** The trained PyTorch model is not loaded anywhere in the app.

### Why the Real Model Is Not in the App

1. No `torch.save()` call was ever added to the training script — no checkpoint file was saved
2. PyTorch with GPU dependencies cannot run on Streamlit Cloud's free tier
3. No model serialization/loading code was written

### What the App Is Good For
- Demonstrating the proposed 3-modality input interface
- Showing the output format (metastasis %, survival estimate, clinical routing)
- As a UI wireframe/mockup for a real deployed system
- The UI design is complete and professional

---

## 9. What Is Done vs. What Is Remaining

### ✅ DONE

| Item | Notes |
|---|---|
| SEER data acquired & cleaned | `seer_rcc_2010_2018_clean.csv` — 36k+ rows |
| TCIA DICOM data downloaded | ~267 patient folders on disk |
| CPTAC genomics data downloaded | Folders exist in `cptac_ccrcc/` |
| EDA on SEER data | Feature distributions, class imbalance documented |
| Full architecture designed & coded | `models/swin3d_fusion.py` — Swin3D + GenomicsTF + CrossAttn |
| GPU training pipeline runs without crashing | On RTX A2000 12GB |
| Class imbalance handled with pos_weight | `BCEWithLogitsLoss(pos_weight=15.7)` |
| Stratified train/val split implemented | Guarantees positive cases in validation |
| 30 epochs trained, real AUC measured | **AUC = 64.1%** |
| Streamlit UI built & deployed | Accessible via public URL |
| GitHub repo organized & pushed | All code, no data/model files |
| This honest documentation | Completed |

### ❌ REMAINING

| Item | Priority | Effort |
|---|---|---|
| **LightGBM on full 36k SEER records** | 🔴 Highest | Low-Medium |
| **True patient ID linkage** (TCGA ↔ CPTAC ↔ clinical labels) | 🔴 High | Very High |
| **Model checkpoint saving** (`torch.save`) | 🟡 Medium | Trivial (1 line) |
| **Real DICOM preprocessing** (SimpleITK, resample, crop) | 🟡 Medium | High |
| **Real RNA-seq data loading** into training | 🟡 Medium | Medium (needs linkage first) |
| **Model inference in Streamlit** (load .pt, run forward pass) | 🟡 Medium | Medium |
| **Benchmark comparison** (LR, RF, XGBoost vs DL) | 🟡 Medium | Medium |
| **Radiomics extraction** from DICOMs (PyRadiomics) | 🟢 Lower | Medium |
| **Statistical significance testing** (DeLong AUC test) | 🟢 Lower | Medium |
| **SHAP explainability** for feature importance | 🟢 Lower | Medium |
| **External validation** on held-out cohort | 🟢 Lower | High |
| **Full hyperparameter search** (Optuna) | 🟢 Lower | Medium |

---

## 10. Main Contributions

Being completely honest about what is genuinely novel vs. what is standard implementation:

### Genuine Contributions
1. **End-to-end multimodal fusion pipeline** for RCC — defining the data flow from 3 heterogeneous cancer data sources into a single Cross-Attention-fused model
2. **Cross-Attention Fusion module** with bidirectional modality interaction (imaging ↔ genomics ↔ clinical)
3. **Multi-task prediction heads** — simultaneous prediction of metastasis probability, survival risk, metastasis site (4 organs), and clinical routing
4. **SEER RCC data cleaning pipeline** — reproducible for future researchers
5. **Multi-source dataset download scripts** — automated acquisition from TCIA and GDC APIs
6. **Honest documentation of the patient linkage problem** — this is itself a contribution; it prevents future researchers from making the same mistake

### What This Is NOT
- This is **not a validated clinical tool**
- The multimodal fusion is **not yet training on aligned multimodal data**
- The imaging and genomics branches **are not contributing to predictions**
- The 64.1% AUC reflects a model learning purely from 7 clinical features on 267 patients

---

## 11. Future Steps

Ordered by impact and feasibility:

### 🔴 Immediate (1–4 weeks)

**Step 1: Train LightGBM on full SEER dataset**
- Use all 36,000 records with 20+ clinical features
- Expected AUC: 0.80–0.88 based on published RCC benchmarks
- Script already started: `scripts/train_lightgbm_baseline.py`
- This will become the primary high-performing model

**Step 2: Save model checkpoints**
- Add `torch.save(model.state_dict(), f'checkpoint_epoch{epoch}.pt')` to training loop
- Save on best validation AUC only

**Step 3: Run real model in Streamlit**
- Load checkpoint, run CPU inference, replace heuristic formula

---

### 🟡 Short-Term (1–3 months)

**Step 4: DICOM preprocessing pipeline**
- `SimpleITK` for resampling CT volumes to uniform 1mm³ spacing
- Tumor ROI cropping using bounding boxes or segmentation
- Enables the imaging branch to receive real data

**Step 5: TCGA ↔ CPTAC patient ID crosswalk**
- Download GDC biospecimen crosswalk file
- Map CPTAC IDs to TCGA sample IDs
- Gives ~80–100 patients with both imaging AND genomics

**Step 6: Real RNA-seq data loading**
- Load actual `.tsv` gene expression values from CPTAC
- Log1p + z-score normalization
- Feed real genomics into Genomics Transformer

**Step 7: Benchmark comparison**
- Compare Logistic Regression, Random Forest, XGBoost, LightGBM against the DL model
- Use DeLong test to compare AUCs with statistical significance

---

### 🟢 Medium-Term (3–6 months)

**Step 8: Radiomics features**
- Extract 100+ shape/texture features from CT using PyRadiomics
- Train LightGBM on radiomics + clinical features
- Expected AUC: 0.82–0.88 based on published studies

**Step 9: True multimodal training**
- With ~80 linked patients (TCGA/CPTAC), train the full 3D Swin + GenomicsTF + CrossAttn
- Pre-train the Swin branch on public CT datasets first (transfer learning)

**Step 10: External validation + publication**
- Validate on a held-out cohort (e.g., TCGA patients not in training)
- Target journals: *European Urology*, *JAMA Network Open*, *Nature Communications*

---

## 12. Project File Structure

```
e:/rcc/
├── Complete_Project_Report.md      ← This honest report
├── README.md                       ← GitHub-facing summary
├── Technical_Details_Truth.md      ← Earlier technical notes
├── requirements.txt                ← Python dependencies
├── streamlit_app.py                ← Deployed UI (heuristic formula, not real model)
├── main.py                         ← Older pipeline entry point
│
├── seer_rcc_2010_2018_clean.csv    ← Cleaned SEER data (36k+ rows)
├── export.txt                      ← Raw SEER fixed-width export (29 MB)
├── export.dic                      ← SEER data dictionary
│
├── TCIA_TCGA-KIRC_09-16-2015 (2)/ ← CT DICOM scans (~267 patients)
│   └── tcga_kirc/
│       ├── TCGA-B0-4699/
│       └── ...
│
├── cptac_ccrcc/                    ← CPTAC genomics folders (~110 patients)
│
├── models/
│   ├── mmrccnet.py                 ← Earlier model variant
│   └── swin3d_fusion.py            ← FULL coded architecture (not trained on real data)
│                                      Contains: SwinTransformer3D, GenomicsTransformer,
│                                      CrossAttentionFusion, SharedDenseLayers, FusedSwin3DNet
│
├── scripts/
│   ├── 01_download_tcia.py         ← TCIA download script
│   ├── 02_download_gdc.py          ← GDC/CPTAC download script
│   ├── final_gpu_train.py          ← ACTUAL TRAINING SCRIPT (30 epochs, 64.1% AUC)
│   ├── train_lightgbm_baseline.py  ← Future LightGBM (not yet run on full data)
│   ├── benchmark_models.py         ← Future benchmarking (not yet run)
│   ├── extract_radiomics.py        ← Radiomics pipeline (not yet run)
│   ├── build_multimodal_manifest.py ← Patient linkage builder (not yet run)
│   ├── optimize_models.py          ← Hyperparameter search (not yet run)
│   ├── run_statistical_validation.py ← Statistical tests (not yet run)
│   ├── train_best_model.py         ← Earlier training attempt
│   ├── train_multimodal.py         ← Earlier multimodal training attempt
│   └── train_swin3d.py             ← Earlier Swin-only training attempt
│
├── data/
│   ├── datasets.py
│   ├── dicom/
│   ├── genomics/
│   ├── manifests/
│   └── radiomics/
│
├── configs/                        ← Model configuration YAML files
├── eda/                            ← EDA outputs
└── runs/                           ← TensorBoard training logs
```

---

## Summary: Prototype vs. Target Reality

| Aspect | Current State | Target State |
|---|---|---|
| Training data size | 267 patients | 36,000+ (SEER) + ~80 linked (TCGA/CPTAC) |
| Imaging input | Zero tensors | Real preprocessed 3D CT volumes |
| Genomics input | Zero tensors | Real RNA-seq gene expression |
| Clinical input | ✅ Real (7 features) | Real (20+ features) |
| Model type | Proxy 3D CNN + GenomicsTF | Full 3D Swin + GenomicsTF |
| **ROC AUC** | **0.641 (real, honest)** | **Target: >0.85** |
| Accuracy (real) | ~90.7% (misleading — class imbalance) | >85% at balanced threshold |
| Prediction basis | Purely 7 clinical features | True multimodal fusion |
| Streamlit output | Heuristic formula | Real model inference |
| Clinical utility | None — proof of concept only | Potential decision support tool |
| Publication ready | No | After external validation |

---

*This report was written with complete transparency. Every number cited reflects either a mathematically computed result from actual training runs or an honest acknowledgement of what was not done. Nothing has been fabricated or embellished.*

*Last updated: June 2026*
