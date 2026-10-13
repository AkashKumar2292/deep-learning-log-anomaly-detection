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

## 🏗️ Architecture & Version 3.0 Methodology

### 1. Chronological Splitting & Leakage Prevention
To prevent temporal sequence leakage, the 4.7M dataset is strictly split chronologically **before** windowing:
- **Train Set (70%)**: 3,299,445 logs used to train the Autoencoder exclusively on normal sequences (`Is_Anomaly = 0`).
- **Validation Set (10%)**: 471,349 logs used for Early Stopping and threshold calibration.
- **Test Set (20%)**: 942,699 unseen logs reserved for final evaluation.

### 2. LSTM Autoencoder (Deep Learning)
- **Embedding Layer**: Vocabulary size $\to$ 16D dense vectors.
- **Encoder (LSTM)**: Compresses a 20-log window into a 32D hidden state vector.
- **Decoder (LSTM)**: Reconstructs the compressed vector back into token probabilities.
- **Training Strategy**: Trained using Cross-Entropy Loss with Adam optimizer ($lr=0.001$, batch size = 512) paired with **Early Stopping** (`patience=3`).

### 3. Threshold Calibration & Sensitivity
Reconstruction loss thresholds are computed using normal validation windows across the 90th, 95th, 97.5th, and 99th percentiles.

---

## 📈 Benchmark Performance (Version 3.0 Evaluation)

### Primary Test Set Performance (95th Percentile Validation Threshold)
| Model / Approach | Precision | Recall | F1-Score | Accuracy | PR-AUC | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Severity Baseline** | 0.0000 | 0.0000 | **0.0000** | 93.60% | — | — |
| **LSTM Autoencoder (DL)** | **0.7389** | **0.1981** | **0.3124** | **94.42%** | **0.2487** | **0.6508** |

### Robustness & Diagnostics (Version 3.0 Highlights)
- **Non-Overlapping Sequence Test (Step = 20)**: Re-evaluating test sequences without overlap confirmed metric stability (**F1-Score: 0.3088**, Precision: 0.7406, Recall: 0.1951).
- **Out-of-Vocabulary (`<UNK>`) Impact**: Diagnostics revealed **85.4%** of test windows contain unseen log templates due to long-term system evolution.
- **Ablation Study**: Excluding `<UNK>` tokens from reconstruction loss drops the F1-Score to **0.2127**, demonstrating that unknown log events actively contribute to the Autoencoder's anomaly signal.

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