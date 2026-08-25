# Sample NIRS Datasets

This directory contains pre-processed sample datasets ready for immediate evaluation in the **NIRS Web App**.

## Available Datasets

### 1. `sample_nirs_data.fif.gz`
- **Modality**: Functional Near-Infrared Spectroscopy (fNIRS / NIRS)
- **Format**: MNE-Python Raw FIF compressed format (`.fif.gz`)
- **Signal Preprocessing Applied**:
  - Temporal Derivative Distribution Repair (**TDDR**) for motion artifact removal
  - Bandpass filtering (0.01 Hz - 0.5 Hz) to remove physiological drift and cardiac noise
  - Optical Density to Hemodynamic concentration conversion (HbO / HbR)
- **Experimental Paradigm & Tasks**:
  - `Finger Sequencing`: Complex fine-motor cognitive task
  - `Simple Tapping`: Periodic motor tapping baseline
  - `Rest`: Baseline resting state
- **Channel Configuration**: Standard 10-20 optical montage over Motor, Premotor, and Prefrontal Cortices.

## How to Use in the Web App

1. Start both the **Backend** and **Frontend** servers (see main [README.md](../README.md)).
2. Navigate to `http://localhost:3000`.
3. Drag & drop `sample_nirs_data.fif.gz` into the upload zone or select it from the pre-loaded files list.
4. Select the activities you want to compare (e.g. `Finger Sequencing` vs `Simple Tapping`).
5. Click **"Analyze Data"** to generate topographic maps, classifier accuracy tables, and connectivity plots.
