# Phishing Email Detection Model

A machine learning model built with Scikit-learn that classifies emails as **Phishing** or **Safe** using NLP techniques and TF-IDF feature extraction.

---

## Features

- Automatically downloads and prepares the UCI SMS Spam Collection dataset
- Extracts email-relevant features: URLs, urgent keywords, text length, digit count, special characters
- Trains and compares two models: Logistic Regression and Random Forest
- Selects the best-performing model automatically
- Generates visualizations: confusion matrix, feature importance chart, label distribution
- Runs live predictions on sample phishing and legitimate emails

---

## Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib

Install dependencies:

```
pip install scikit-learn pandas numpy matplotlib
```

---

## Usage

```
python phishing_email_detection.py
```

The script will:
1. Download the dataset automatically (first run only)
2. Train both models and print accuracy + classification report
3. Save charts to the current directory
4. Run predictions on 6 sample emails

---

## Output Files

| File | Description |
|---|---|
| `confusion_matrix.png` | True/False positive breakdown for the best model |
| `feature_importance.png` | Top words that indicate Phishing vs Safe |
| `label_distribution.png` | Class balance in the dataset |

---

## Dataset

[UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) — 5,574 labeled messages mapped to:
- `spam` → **Phishing**
- `ham` → **Safe**

---

## Model Performance

Expected accuracy: **~97–98%** on the test set.

---

## Project Structure

```
├── phishing_email_detection.py   # Main script
├── confusion_matrix.png          # Generated on run
├── feature_importance.png        # Generated on run
├── label_distribution.png        # Generated on run
└── data/                         # Dataset downloaded here
```

---

## Internship Project

Built as part of a cybersecurity internship project focused on email threat detection using classical machine learning.
