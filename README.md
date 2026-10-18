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

## 🏗️ Methodology & Architecture (Version 7.0)

### 1. Chronological Splitting & OOV Diagnostics
Data is split chronologically **before** windowing to eliminate lookahead bias (70% Train, 10% Val, 20% Test).
- **Severe Concept Drift**: Test set token OOV rate is **75.44%**. **85.4%** of all test windows contain at least one `<UNK>` token.

### 2. OOV-Robust Denoising LSTM Autoencoder
To combat the artificial inflation of reconstruction errors caused by unseen templates, an **OOV-Robust model** is introduced. It is trained on normal sequences only, but the input is actively corrupted by randomly masking 15% of known tokens as `<UNK>` (`0`), forcing the model to reconstruct the original clean sequence and learn contextual immunity to drift.

### 3. Synthetic Stress Testing
A pure, OOV-free "Synthetic Substrate" is extracted from the normal test data. Specific perturbations are systematically injected at 10%, 20%, and 40% corruption fractions to evaluate robustness:
- `substitute`: Random known training events.
- `rare`: Rarely occurring known events.
- `oov`: Unseen `<UNK>` tokens.
- `repeat`: Repeating the previous event.
- `shuffle`: Order permutations.

---

## 📈 Benchmark Performance (Version 7.0 Evaluation)

All models utilize a 99th-percentile threshold calibrated on the normal validation set. 

### Primary Full BGL Test Evaluation
| Model | AUC | AP | Accuracy | Precision | Recall | F1-Score | FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard LSTM AE** | 0.5913 | **0.1781** | **94.34%** | **1.0000** | 0.1166 | **0.2088** | **0.0000** |
| **OOV-Robust LSTM AE** | **0.6578** | 0.1260 | 92.62% | 0.1919 | **0.0477** | 0.0764 | 0.0137 |
| **Bigram Transition** | 0.5784 | 0.0776 | 93.61% | 0.6567 | 0.0036 | 0.0072 | 0.0001 |
| **Event Rarity** | 0.5674 | 0.0730 | 93.60% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Count Z-Score** | 0.5387 | 0.0684 | 93.60% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Secondary Disjoint Real Test (Non-Overlapping)
| Model | AUC | AP | Accuracy | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Standard LSTM AE** | 0.5298 | 0.3374 | 78.25% | **0.2301** |
| **OOV-Robust LSTM AE** | **0.5992** | **0.3461** | 74.90% | 0.0839 |
| **Classical Baselines** | ~0.4900 | ~0.2400 | 75.00% | 0.0000 |

*Note: While the OOV-Robust model improves overall discriminative power (AUC), the strict 99th-percentile threshold restricts its F1-score due to precision-recall trade-offs. The Deep Learning models significantly outperform the classical statistical baselines.*

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