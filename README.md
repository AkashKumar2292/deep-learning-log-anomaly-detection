# Deep Learning-Based Anomaly Detection in Cloud-Native Applications

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end implementation comparing an unsupervised **Deep Learning approach (LSTM Autoencoder)** against a **Rule-Based Severity Baseline** for detecting anomalies in cloud application system logs using the BGL dataset.

---

## 📌 Problem Statement
Cloud-native applications generate vast volumes of unstructured log data. Traditional threshold-based monitoring relies on static keyword matching (e.g., searching for `ERROR` or `FATAL`), which struggles to detect complex sequential anomaly patterns and frequently triggers high false-positive rates. This project implements an unsupervised **LSTM Autoencoder** to learn normal log sequence patterns and flags reconstruction errors as anomalies, benchmarking its accuracy against standard rule-based filters.

---

## 📊 Dataset & Preprocessing Pipeline
- **Dataset**: [BGL (BlueGene/L Supercomputer Log)](https://github.com/logpai/loghub)
- **Full Scale**: **4,713,493 log lines** (~348,460 ground-truth anomalies)
- **Data Pipeline**:
  1. **Regex Parsing**: Structured extraction of headers and raw content.
  2. **Dynamic Variable Masking**: Replaces memory addresses, IPs, numbers, and node IDs with `<*>` placeholders to form **Event Templates**.
  3. **Train-Only Vocabulary Mapping**: Integer tokens derived strictly from training data. Unseen templates in validation/testing map to `<UNK>` (`0`).
  4. **Sliding Windows**: Sequence generation (Window size = 20 logs, Step size = 5 for primary; Step size = 20 for non-overlapping validation).

---

## 🏗️ Version 5.0 Methodology & Diagnostics

### 1. Chronological Splitting & Leakage Prevention
To prevent temporal sequence leakage, the dataset is split chronologically **before** sequence generation:
- **Train Set (70%)**: 3,299,445 logs used to train the Autoencoder exclusively on normal sequences (`Is_Anomaly = 0`).
- **Validation Set (10%)**: 471,349 logs used for Early Stopping and threshold calibration.
- **Test Set (20%)**: 942,699 unseen logs reserved for evaluation.

### 2. Comprehensive Vocabulary Drift & Diagnostic Analysis
- **Token-Level Out-of-Vocabulary (OOV)**: **75.44%** of test log tokens are unseen in the training split.
- **Window-Level OOV**: **80.93%** of test sequence windows contain at least one unseen `<UNK>` token.
- **All-OOV Windows**: **70.83%** of test windows consist entirely of unseen `<UNK>` tokens due to system runtime changes over time.

### 3. Training & Dual Baseline Strategy
- **LSTM Autoencoder**: Embedding (16D), Encoder/Decoder LSTM (32D), trained with Adam ($lr=0.001$, batch size = 512) and Early Stopping (`patience=3`).
- **Baseline 1 (FPR-Constrained Rule)**: Calibrated on validation data to enforce Max FPR $\le$ 0.05 (threshold: $>20$ critical events per window).
- **Baseline 2 (Validation-Optimized Severity Rule)**: Optimized directly for validation F1-score (threshold: $>1$ critical event per window).

---

## 📈 Benchmark Performance (Version 5.0 Evaluation)

### Primary Test Set Evaluation
| Model / Approach | Accuracy | Precision | Recall | F1-Score | PR-AUC | ROC-AUC | FPR | FNR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: FPR-Constrained Rule (>20)** | 93.60% | 0.0000 | 0.0000 | **0.0000** | — | — | 0.0000 | 1.0000 |
| **Baseline 2: Val-Optimized Severity (>1)** | 89.51% | 0.3705 | 0.7816 | **0.7045** | — | — | 0.0971 | 0.2184 |
| **LSTM Autoencoder (95th Percentile)** | **93.45%** | **0.4590** | **0.1248** | **0.1963** | **0.1809** | **0.6146** | 0.0101 | 0.8752 |

### Robustness & Masked Ablation Results
- **Non-Overlapping Sequence Test (Step = 20)**: Evaluating non-overlapping test windows confirmed stable metrics (**F1-Score: 0.1968**, Precision: 0.4609, Recall: 0.1251, PR-AUC: 0.1814).
- **`<UNK>`-Excluded Masked Ablation**: Masking out `<UNK>` tokens during window-level loss averaging improved the Autoencoder F1-score from **0.1963** to **0.2970**.

---

## 📂 Repository Structure

```text
deep-learning-log-anomaly-detection/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── notebooks/
│   └── Log_Anomaly_Detection.ipynb   # Complete prototype (2k) & full (4.7M) execution pipeline

```

---

## 🚀 How to Run

### 1. Clone the Repository

``` Bash
git clone [https://github.com/AkashKumar2292/deep-learning-log-anomaly-detection.git](https://github.com/AkashKumar2292/deep-learning-log-anomaly-detection.git)
cd deep-learning-log-anomaly-detection
```
### 2. Set Up Virtual Environment

``` Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```
### 3. Run the Jupyter Notebook

``` Bash
jupyter notebook notebooks/Log_Anomaly_Detection.ipynb
```
### 4. Running on Google Colab
1. Upload notebooks/Log_Anomaly_Detection.ipynb to Google Colab.
2. Mount your Google Drive containing BGL.log.zip or upload the dataset directly.
3. Enable GPU Hardware Acceleration (Runtime > Change runtime type > T4 GPU).
4. Execute cells sequentially.

## 📜 License
Distributed under the MIT License.