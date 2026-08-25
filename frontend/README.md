# 💻 NIRS Web App Frontend

Modern, interactive React single-page application (SPA) providing neuroscientists and BCI researchers with a real-time analytics dashboard for Functional Near-Infrared Spectroscopy (fNIRS) datasets.

---

## ✨ Highlights & Capabilities

- 📤 **Drag & Drop Data Ingestion**: Seamlessly upload and validate `.fif` / `.fif.gz` NIRS recordings.
- 🎯 **Interactive Paradigm & Activity Selection**: Select specific cognitive tasks (`Finger Sequencing`, `Tapping`, `Mental Arithmetic`, `Rest`) for comparison.
- 📊 **Multi-Panel Analytics Dashboard**:
  - Topographic brain region importance mapping
  - Inter-channel functional connectivity correlation matrices
  - Multi-classifier benchmark metrics (Accuracy, ROC-AUC, Confusion Matrix)
  - Chronological temporal validation charts
  - Automated neurophysiological interpretation reports
- 📱 **Responsive & Clean UI**: Styled with modern CSS components and real-time loading state indicators.

---

## 🛠️ Tech Stack

- **Framework**: React 18
- **Data Communication**: Fetch API with CORS preflight handling
- **Styling**: Modular CSS3 with custom visualizer components

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Navigate to frontend directory
cd frontend

# Install packages
npm install
```

### 2. Start the Development Server

Make sure the backend API is running on `http://localhost:5000`.

```bash
npm start
```

Your browser will automatically open: **`http://localhost:3000`**.

---

## 📦 Production Build

To compile static assets for production deployment:

```bash
npm run build
```
