# MH6822 Regulatory Technology — Assignment 1

**Name:** SHEN YUANJING
**Matriculation ID:** G2505264L
**Email:** YUANJING001@e.ntu.edu.sg

---

## Assignment Overview

- **Regulated Entity:** JPMorgan Chase & Co.
- **Jurisdictions:** United States (OCC Bulletin 2026-13) + European Union (EU AI Act 2024)
- **Domain:** AI Governance — Credit Decision Model Risk & Fairness Monitoring
- **Option:** Option C — Analytical Design with Quantitative Component

---

## File Index

| File | Format | Description |
|------|--------|-------------|
| `README.md` | Markdown | This file |
| `Task1_Research.docx` | Word | Entity selection, regulatory landscape, references |
| `Task2_ValuesAudit.docx` | Word | Four-question values audit |
| `Task3_ToolDesign.docx` | Word | Tool architecture, jurisdiction config layer, failure modes, model card |
| `Task3_Analysis.py` | Python | Synthetic data, fairness metrics (DPD/EOD), PSI drift, dual-jurisdiction reports, sensitivity analysis |
| `Task3_Presentation.pptx` | PowerPoint | 8-slide senior management pitch deck |
| `MP3_Narration.mp3` | MP3 | ~4-minute audio walkthrough of submission |

---

## How to Run the Python Analysis

```bash
pip install scikit-learn pandas numpy
python Task3_Analysis.py
```

No external data files required — the script generates synthetic data internally.

---

## Data Collaboration

The synthetic dataset in `Task3_Analysis.py` was generated independently.  
*(Update this line if you collaborated with classmates on data.)*

---

## Key Findings Summary

The same logistic regression credit model produces **opposite compliance outcomes** depending on jurisdiction:

| Metric | Value | US (OCC 2026-13) | EU (AI Act 2024) |
|--------|-------|-------------------|------------------|
| DPD | 0.385 | Advisory only — no breach | **FAIL** — exceeds 0.05 threshold |
| PSI | 0.034 | GREEN — stable | N/A — post-market plan applies |
| EOD | 0.099 | Advisory only | **FAIL** — exceeds 0.05 threshold |

This divergence — not detectable without jurisdiction-aware tooling — is the core problem the tool addresses.
