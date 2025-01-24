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
3. **Support Vector Machine**: Detects span using an SVM model form sklearn.

---

## File Structure

### **classifiers/**
Contains implementations of supervised baselines for hallucination detection.

- **`mbert_token_classifier_...`**: Saved models from previous iterations of the mBERT token classification baseline.
- **`NN_Classifier.py`**: Main file for the neural network-based mBERT token classification baseline.
- **`SVM_Classifier.py`**: Implementation of an SVM-based baseline.
- **`NN_utils.py`**: Utility functions for models (e.g., metrics, data processing).
- **`NN_experiments_qualitative_analysis.ipynb`**: Qualitative analysis of the NN classifier results.

---

### **classifiers/GPT/**
Contains code and data related to the GPT baseline.

- **`gpt_testing.py`**: Script to send query-response pairs to the GPT API.
- **`calc_accuracy.py`**: Evaluation of the zero-shot GPT with metrics.
- **`hallucinations.json`**: Output data identifying hallucinated tokens.

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

### **preprocess/**
Scripts and documentation for preprocessing data.

- **`preprocess.py`**: Main script for preprocessing raw datasets.
- **`load_data.py`**: Utilities to load and prepare data for model training.
- **`preprocessing.md`**: Documentation describing the preprocessing pipeline.

---

### **Root Files**
- **`NLP_Milestone_2.pdf`**: Report for Milestone 2.
- **`Management Summary`**: Management summary of the project.
- **`README.md`**: This file, explaining the project and file structure.
- **`requirements.txt`**: List of required Python libraries.

---

## How to run the project

1. Clone the repository:
   ```bash
   git clone https://github.com/nikoczernin/MuSHROOM
   cd MuSHROOM
   pip install -r requirements.txt
   ```
2. Run the classifiers in the classifiers folder in the root
   - Each classifier can be run in its respective .py file, but for the NN it is recommended to run from the .ipynb
   - To tweak run parameters see the instance of the Args class, which defines the params for the model scope
   - See the logged results
   - For neural network training a GPU is heavily recommended

3.  Qualitative analysis:
   - Run the NN_experiments_qualitative_analysis.ipynb
   - The experiments are run there separately too.