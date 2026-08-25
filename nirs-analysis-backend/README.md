# 🧠 NIRS Analysis Backend

A high-performance Flask REST API service powering signal processing, artifact correction, and machine learning pipelines for Functional Near-Infrared Spectroscopy (fNIRS) neuroimaging datasets.

---

## ⚡ Tech Stack & Libraries

- **Framework**: [Flask](https://flask.palletsprojects.com/) + Flask-CORS
- **Neuroimaging Engine**: [MNE-Python](https://mne.tools/) (fNIRS signal processing, channel montages, MBLL conversion)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/), [imbalanced-learn](https://imbalanced-learn.org/) (SMOTE, K-Fold CV, Ensemble Classifiers)
- **Signal Analysis**: [SciPy](https://scipy.org/), [PyWavelets](https://pywavelets.readthedocs.io/) (wavelet decomposition, PSD, TDDR)
- **Visualization**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) (Base64-encoded real-time plot generation)

---

## 📁 Directory Architecture

```plaintext
nirs-analysis-backend/
├── app/
│   ├── api/
│   │   ├── routes.py          # API endpoints (upload, file list, analyze, temporal validation)
│   │   └── validators.py      # Input validation & schema checks
│   ├── core/
│   │   ├── nirs_processor.py  # MNE data loading, filtering, connectivity heatmaps, PSD
│   │   ├── nirs_ml.py         # Multi-model classification, SMOTE, temporal bias validation
│   │   └── visualizer.py      # Brain region mapping & topographic plots
│   ├── models/                # Data structures & result models
│   ├── utils/                 # JSON serialization & file handlers
│   ├── config.py              # Application settings & environment variables
│   └── __init__.py            # Flask Application Factory
├── tests/
│   ├── test_api.py            # API route integration tests
│   └── test_analyzer.py       # Core utility & processing unit tests
├── uploads/                   # Runtime storage for uploaded NIRS files
├── requirements.txt           # Python dependency specifications
└── run.py                     # Backend entry point
```

---

## 🚀 Quick Start Guide

### 1. Create and Activate Virtual Environment

```bash
# Navigate to backend directory
cd nirs-analysis-backend

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Backend Server

```bash
python run.py
```
> The API server will start on **`http://localhost:5000`**.

---

## 📡 API Endpoints

| Method | Route | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API status and endpoint directory |
| `GET` | `/api/health` | Service health check and system information |
| `POST` | `/api/upload` | Upload `.fif` / `.fif.gz` NIRS recordings |
| `GET` | `/api/files` | List all available datasets with channel & duration metadata |
| `GET` | `/api/available_activities` | Extract annotated experimental conditions/tasks |
| `POST` | `/api/analyze` | Execute complete preprocessing, feature extraction & ML classification |
| `POST` | `/api/temporal_validation` | Test for chronological signal leakage & temporal bias |

---

## 🧪 Running Tests

```bash
pytest
```
