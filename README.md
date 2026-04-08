# Exoplanet Detection via Transit Photometry

---
![demo](notebooks/drafts/transit.gif)
---

## Overview

This project detects exoplanets by analyzing stellar brightness time series collected through **Transit Photometry** — a technique where a planet crossing in front of its host star causes a measurable dip in light intensity (~0.01%).

The core research question: **does classical signal processing outperform deep learning on extremely low signal-to-noise astronomical data?**

---

## Key Results

| Model | Approach | Recall | Accuracy |
|---|---|---|---|
| Random Forest + FFT | Manual feature engineering | **0.93** | **0.94** |
| 1D-CNN | End-to-end deep learning | 0.51 | 0.52 |

**Finding:** Translating the raw light curve from the time domain into the frequency domain via Fast Fourier Transform allowed the classical model to isolate periodic transit patterns effectively. The 1D-CNN, despite applying Savitzky-Golay detrending, batch normalization, and z-score normalization, could not extract meaningful structure from the raw signal without the mathematical preprocessing step.

> Proper signal processing proved more effective than architectural complexity.

---

## Dataset

- **Source:** [Exoplanet Detection Dataset](https://www.kaggle.com/datasets/ronaldkroening/exoplanet-detection-dataset) via Kaggle
- **Scale:** 5,899 stars · 4,608 time steps · binary classification (planet / no planet)
- **Preprocessed data & model weights:** [Google Drive](https://drive.google.com/drive/folders/1Bq1bLAs5mcZg8LKSCG_P3ACppRvxIGIx?usp=sharing)

---

## Pipeline

### 1. Data Preprocessing — `01_data_cleaning.ipynb`
- Linear interpolation to fill missing values
- Alignment of all time series to a uniform length of 4,608 steps

### 2. Feature Engineering — `02_feature_engineering_fft.ipynb`
- Savitzky-Golay filter to remove low-frequency stellar variability (detrending)
- Fast Fourier Transform (FFT) to generate frequency-domain power spectrum features

### 3. Modeling & Evaluation — `03_baseline_model.ipynb` · `04_deep_learning_cnn.ipynb`
- `RandomForestClassifier` trained on extracted FFT features
- `1D-CNN` (PyTorch) trained on raw and detrended time series
- Evaluation via ROC-AUC and Confusion Matrix

---

## Project Structure

```
exoplanet-hunting/
├── data/
│   ├── raw/                  # Source CSV files
│   └── processed/            # Cleaned time series and FFT features
├── models/                   # Saved model weights (e.g. rf_baseline.pkl)
├── notebooks/                # Jupyter notebooks for each pipeline stage (01–04)
├── src/
│   ├── preprocessing.py      # Interpolation and detrending logic
│   └── features.py           # FFT feature extraction
├── train.py                  # Entry point for training the baseline model
└── README.md
```

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/ZED-or-ZERO/Exoplanet_Hunting.git
cd Exoplanet_Hunting

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the baseline model
python train.py
```