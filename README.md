# NeuroScan — Brain Tumor Classification Using CNN

An educational deep-learning project that classifies brain MRI images into four dataset categories using a custom Convolutional Neural Network (CNN).

> **Important:** This is a student demonstration, not a medical device. Predictions must not be used to diagnose, rule out, or treat a medical condition.

## Features
- Image-folder dataset loading with an automatic training/validation split
- Custom CNN with three convolutional blocks
- Rescaling, max pooling, dropout, and softmax output
- Early stopping and best-model checkpointing
- Optional test-folder evaluation
- Streamlit interface for image upload and prediction

## Tech stack
Python, TensorFlow/Keras, Streamlit, NumPy, Pillow, scikit-learn, Matplotlib.

## Dataset setup
Use a properly licensed public brain MRI classification dataset with four categories. One commonly used source is the Kaggle dataset titled **Brain Tumor MRI Dataset**; check the dataset's current license and terms before use.

Place the extracted folders like this:

```text
NeuroScan_CNN_Project/
├── app.py
├── train_model.py
├── requirements.txt
├── data/
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/                 # optional, recommended
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
└── models/
```

Folder names must be subfolders of the training/testing directories. TensorFlow assigns class IDs alphabetically based on folder names; the app currently displays a fixed class list, so verify that the dataset's class order is `glioma`, `meningioma`, `notumor`, `pituitary`. If the printed class order differs, update `CLASS_NAMES` in `app.py` to match `models/class_names.txt`.

## Run locally

### 1. Create and activate a virtual environment (recommended)
Windows:
```powershell
py -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train
Download/extract the dataset into `data/`, then run:
```bash
python train_model.py
```

This saves the trained model to `models/brain_tumor_cnn.keras`.

### 4. Launch the app
```bash
streamlit run app.py
```

## Model overview
Input: RGB MRI image resized to 150 × 150 pixels. The model rescales pixel values to 0–1, applies Conv2D + ReLU and max-pooling layers to learn spatial features, then uses a dense layer and dropout before a four-class softmax output.

Loss: sparse categorical cross-entropy. Optimizer: Adam. Main metric: accuracy. Early stopping monitors validation loss.

## Evaluation and limitations
- Report test-set results only after training and evaluating on a separate test set.
- Check precision, recall, F1-score, and a confusion matrix before making performance claims.
- MRI image datasets may contain duplicates, acquisition artifacts, and source-specific patterns. High accuracy does not prove clinical reliability.
- The app's displayed score is the model's softmax output, not a calibrated probability of disease.
- This baseline is intended for learning and demonstration only.
