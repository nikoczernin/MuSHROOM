# MuSHROOM: Hallucination Span Detection Project
NLP class task: Detecting hallucination spans within the Mu-SHROOM text dataset.

## Table of Contents
<!-- TOC -->
* [MuSHROOM: Hallucination Span Detection Project](#mushroom--hallucination-span-detection-project)
  * [Table of Contents](#table-of-contents)
  * [Project Description](#project-description)
    * [Key Approaches:](#key-approaches-)
  * [File Structure](#file-structure)
    * [**baseline/**](#baseline)
    * [**data/**](#data)
    * [**GPT/**](#gpt)
    * [**preprocess/**](#preprocess)
    * [**Root Files**](#root-files)
  * [How to prep the Project](#how-to-prep-the-project)
<!-- TOC -->

---

## Project Description

This project focuses on **hallucination detection** in model-generated responses. The objective is to identify tokens in model outputs that are factually incorrect or inconsistent with the provided query. 

### Key Approaches:
1. **Supervised Baseline (mBERT)**: Fine-tunes mBERT for token classification to detect hallucinated spans.
2. **Unsupervised Baseline (GPT API)**: Sends query-response pairs to the GPT API and analyzes hallucinations based on GPT outputs.

---

## File Structure

### **baseline/**
Contains implementations of supervised baselines for hallucination detection.

- **`mbert_token_classifier_...`**: Saved models from previou iterations of the mBERT token classification baseline.
- **`baseline_NN_mBERT.py`**: Main file for the neural network-based mBERT token classification baseline.
- **`baseline_svm.py`**: Implementation of an SVM-based baseline (not quite fleshed out).
- **`baseline_utils.py`**: Utility functions for baselines (e.g., metrics, data processing).

---

### **data/**
Holds datasets and preprocessed files for training, validation, and exploration.

- **output/**
  - **`val_predictions_mbert2.csv`**: Model predictions.

- **preprocessed/**
  - **`sample_preprocessed.json`**: Sample data prepared for quick testing.
  - **`val_preprocessed.json`**: Validation data used for evaluation.

- **sample/**, **train/**, **val/**: Original datasets.

---

### **GPT/**
Contains code and data related to the GPT baseline.

- **`gpt_testing.py`**: Script to send query-response pairs to the GPT API.
- **`hallucinations.json`**: Output data identifying hallucinated tokens.

---

### **preprocess/**
Scripts and documentation for preprocessing data.

- **`preprocess.py`**: Main script for preprocessing raw datasets.
- **`load_data.py`**: Utilities to load and prepare data for model training.
- **`preprocessing.md`**: Documentation describing the preprocessing pipeline.

---

### **Root Files**
- **`main.py`**: Entry point to run the project, combining preprocessing, training, and evaluation.
- **`NLP_IE_2024WS_Exercise.pdf`**: Project or exercise description file.
- **`README.md`**: This file, explaining the project and file structure.
- **`requirements.txt`**: List of required Python libraries.

---

## How to prep the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/nikoczernin/MuSHROOM
   cd MuSHROOM
   pip install -r requirements.txt
   ```
