# NIRS Conversion & Preprocessing Notebooks

This directory contains research and data preparation Jupyter notebooks used to convert raw multi-format NIRS signals into MNE-Python compatible `.fif.gz` structures.

## Available Notebooks

### 📓 `NIRS_TO_FIF_GZ.ipynb`
- **Objective**: Converts raw optical measurements (NIRx, SNIRF, CSV/text) to standard MNE `Raw` objects.
- **Key Pipeline Steps**:
  1. **Raw Ingestion**: Reads light intensities at 760nm and 850nm wavelengths.
  2. **Optode & Montage Registration**: Maps source-detector pairs to the 10-20 standard topographic head model.
  3. **Modified Beer-Lambert Law (MBLL)**: Converts optical attenuation changes into Oxygenated (\(\Delta[\text{HbO}]\)) and Deoxygenated (\(\Delta[\text{HbR}]\)) hemoglobin concentrations.
  4. **Event Annotation**: Parses trigger channels and assigns descriptive labels (`Finger Sequencing`, `Simple Tapping`, etc.) with onset timestamps and durations.
  5. **Export**: Saves compressed `.fif.gz` files optimized for streaming and web processing.
