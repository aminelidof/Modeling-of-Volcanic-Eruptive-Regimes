# Data-Driven Physics-Informed Modeling of Volcanic Eruptive Regimes: Insights from the Pernicana Fault Seismicity at Mount Etna

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-100%25-brightgreen.svg)]()

This repository contains the complete implementation, numerical simulations, and figure generation pipeline for the scientific study on **Data-Driven Magmatic Transport and Eruptive Decision Assessment at Mount Etna**.

---

## 📌 Authors & Affiliations

* **Mohammed El Amine Fodil**<sup>1,2,*</sup> (Corresponding Author) — `fodilmedam@gmail.com`
* **Asma Khelassi Sefaoui**<sup>1</sup>
* **Merwan Abdelbari**<sup>3</sup>
* **Meriem Fodil**<sup>3</sup>

<sup>1</sup> *Department of Hydraulics, Maghnia University Centre, Tlemcen – Algeria*  
<sup>2</sup> *Laboratoire Ingénierie et Sciences Appliquées IScApp, Maghnia University Centre, Tlemcen - Algeria*  
<sup>3</sup> *Department of Mechanics, Hassiba Ben Bouali University, Chlef – Algeria*  

---

## 🔬 Overview

This codebase implements a two-stage data-driven physics-informed pipeline designed to simulate magmatic conduit transport, seismic response, and eruptive decision analysis at Mount Etna:

1. **Step 1: Hydro-Magmatic Transport Model** — Solves fluid dynamic fields ($P, U, T, \mu, \alpha$) along the main volcanic conduit.
2. **Step 2: Seismic & Structural Response** — Models mechanical stress propagation and seismic rate responses ($N_{eq}, E_{cum}, \sigma_n, \tau_m$).
3. **Eruptive Decision & Audit Engine** — Produces automated JSON audits and publication-ready figures for decision-making and risk assessment.

---

## 📁 Repository Structure

```text
.
├── LICENSE
├── README.md
├── requirements.txt
├── src/
│   ├── app_magma.py                    # Main physics solver & training script
│   └── total_figures.py                # Automated figure generator (Figs 1–16) & JSON exporter
└── data/
    ├── eruptive_decision.json          # Machine-readable decision report
    ├── etna_simulation_results.csv     # Exported 2D spatial-temporal fields
    ├── step1_validation_data.csv       # Benchmark validation metrics
    ├── train_log_step1.csv             # Step 1 PINN/Solver loss history
    ├── train_log_step2.csv             # Step 2 Loss history
    └── etna_all_earthquakes_observations.csv # Seismic observation catalog (225 events)
```

---

## 🚀 Getting Started

### 1. Prerequisites

Make sure you have Python 3.9+ installed. Clone this repository and navigate to the project directory:

```bash
git clone https://github.com/your-username/mount-etna-data-driven.git
cd mount-etna-data-driven
```

### 2. Environment Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

*(Key libraries: `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `torch` or `tensorflow` depending on your backend)*

---

## 💻 Usage & Execution Workflow

### Step 1: Run Physics Simulation and Training

Execute the main solver to run the physics models and produce spatial-temporal fields:

```bash
python src/app_magma.py
```

### Step 2: Generate Figures & Eruptive Audit Report

Run `total_figures.py` to process the simulation logs and generate all publication-ready figures along with the `eruptive_decision.json` audit file:

```bash
python src/total_figures.py
```

All generated figures (Figures 1 through 16) will be saved in high resolution (`.png` / `.pdf`) in the working directory or an `output/` folder.

---

## 📊 Outputs & Eruptive Decision JSON

The pipeline generates `eruptive_decision.json`, structured as follows:

```json
{
  "eruptive_risk_assessment": {
    "status": "Eruptive Alert / Critical Level",
    "critical_threshold_exceeded": true,
    "confidence_score": 0.94
  },
  "metrics": {
    "peak_pressure_MPa": 142.5,
    "max_seismic_rate": 87.2,
    "mne_validation_score": 0.0182
  }
}
```

---

## 📜 Citation

If you use this code, data, or methodology in your research, please cite our paper:

```bibtex
@article{Mohammed2026Etna,
  title={Data-Driven Magmatic Transport and Eruptive Decision Assessment at Mount Etna},
  author={Mohammed, [Last Name] and Asma, [Last Name] and Merwan, [Last Name] and Meriem, [Last Name]},
  journal={[Journal Name]},
  year={2026},
  note={Under Review}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
