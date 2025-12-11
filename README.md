# Fire Flow — EDA & Model Training

This repository contains the exploratory data analysis (EDA) scripts and model training pipeline for **Fire Flow**, a wildfire next-day spread prediction system.  
It builds on the processed multi-band GeoTIFF dataset produced by the `data_processing` pipeline and trains deep learning models for fire spread prediction.

---

## Project Overview

This repo includes:

- Exploratory data analysis (visualization, band inspection, class distribution)
- Data loading utilities for multi-band GeoTIFF wildfire patches
- Model architectures (baseline CNN, U-Net variants, transfer learning experiments)
- Training pipeline + validation metrics
- Logging utilities and saved model experiments

The goal is to evaluate multiple model architectures and identify the best-performing approach for next-day wildfire spread prediction.

---

## Tech Stack

| Category | Technology |
|---------|------------|
| Machine Learning | TensorFlow / Keras |
| Data Processing | rasterio, NumPy, pandas |
| Visualization | matplotlib, seaborn |
| Development | Python 3.9+ |
| Training Platform | Colab / Kaggle / local GPU |

---

## Repository Structure

eda_and_training/
├── eda/
│   ├── band_visualization.ipynb
│   ├── class_distribution.ipynb
│   ├── spatial_examples.ipynb
│   └── temporal_analysis.ipynb
│
├── loaders/
│   ├── dataset_loader.py
│   └── augmentations.py
│
├── models/
│   ├── cnn_baseline.py
│   ├── unet.py
│   └── transfer_unet.py
│
├── training/
│   ├── train.py
│   ├── metrics.py
│   ├── loss_functions.py
│   └── callbacks.py
│
├── experiments/
│   ├── experiment_1_cnn/
│   ├── experiment_2_unet/
│   └── experiment_3_transfer/
│
├── requirements.txt
└── README.md


---

## Data Requirements

This repository expects data in the format produced by:
https://github.com/Fire-Flow-Detection/data_processing

Specifically:

- Each sample is a **32×32×18** multi-band GeoTIFF
- Each label is a **32×32×1** fire mask for the next day
- Files are stored in:

train/
val/
test/

Example directory structure:

processed_data/
├── train/
│ ├── sample_00001.tif
│ ├── sample_00002.tif
│ └── ...
├── val/
└── test/


---

## Exploratory Data Analysis (EDA)

The `eda/` folder includes notebooks for:

- Visualizing each spectral band  
- Inspecting vegetation, slope, NDVI, drought indices  
- Checking fire-mask label imbalance  
- Verifying spatial alignment of all 18 bands  
- Examining sample patches over time  

Run any notebook using Jupyter or Colab.

Example:

```bash
jupyter notebook eda/band_visualization.ipynb
```

### Model Training Pipeline
1. Install dependencies
pip install -r requirements.txt

2. Update dataset path in train.py

Example:

DATA_ROOT = "/content/processed_data"

3. Run training
python training/train.py \
    --model unet \
    --epochs 20 \
    --batch_size 32

4. Outputs

Trained weights (checkpoints/…)

Training logs (IoU, Dice, accuracy)

Comparison plots for all experiments

### Model Architectures
* Baseline CNN

1.Fast to train
2.Lower accuracy, used for benchmarking

* U-Net

1.Encoder-decoder segmentation model
2.Higher IoU and Dice scores
3.Best-performing model in experiments

* Transfer Learning U-Net

1.Pretrained encoder on satellite imagery
2.Improved generalization on small datasets

### Metrics

This project evaluates:

* IoU
* Dice Score
* Binary accuracy
* Precision & Recall

Custom loss functions include:

* Dice Loss
* BCE + Dice hybrid

## Acknowledgements

This training pipeline is part of the Fire Flow wildfire prediction system, developed for academic research and advanced machine learning experimentation.
