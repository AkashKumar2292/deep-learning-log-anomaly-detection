# Deep Learning-Based Anomaly Detection in Cloud-Native Applications

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end implementation comparing an unsupervised **Deep Learning approach (LSTM Autoencoder)** against a **Rule-Based Severity Baseline** for detecting anomalies in cloud application system logs using the BGL dataset.

---

## 📌 Problem Statement
Cloud-native applications generate vast volumes of unstructured log data. Traditional threshold-based monitoring relies on static keyword matching (e.g., searching for `ERROR` or `FATAL`), which struggles to detect complex sequential anomaly patterns and frequently triggers high false-positive rates . This project implements unsupervised **LSTM Autoencoder** architectures designed to learn normal log sequence patterns and flag reconstruction errors as anomalies, benchmarking performance against specialized OOV-robust variants and classical baselines.

---

## 📊 Dataset & Preprocessing Pipeline
- **Dataset**: [BGL (BlueGene/L Supercomputer Log)]
- **Scale & Class Distribution**: ~4.7 million parsed log lines with a highly imbalanced log-level distribution (92.6% normal logs vs. 7.4% anomalous logs).
- **Data Pipeline**:
  1. **Regex Parsing**: Custom regular expression parsing extracts structured headers and raw content, converting variable fields (e.g., IP addresses, core node IDs, memory hex codes) into generic `<*>` placeholders to form standardized **Event Templates**.
  2. **Vocabulary Mapping**: The training vocabulary encompasses 30,050 unique event templates, including a dedicated `<UNK>` token (Index 0) reserved for unseen events.
  3. **Chronological Splitting**: Data is split chronologically prior to windowing into 70% Training, 10% Validation, and 20% Testing sets to prevent lookahead bias.
  4. **Sliding Windows**: Sequence generation uses a sliding window size of 20. Overlapping sequences utilize a step size of 5, while non-overlapping test evaluation utilizes a step size of 20.

---

## 🏗️ Methodology & Models (Version 9.0)

Version 9.0 evaluates six distinct anomaly detection strategies:

1. **Standard LSTM Autoencoder**: A PyTorch-based sequence autoencoder utilizing an embedding dimension of 16 and a hidden dimension of 32.
2. **OOV-Robust Denoising LSTM Autoencoder**: Shares the architecture of the standard LSTM, but is trained on normal sequences artificially corrupted with a 15% random `<UNK>` masking probability to force contextual reconstruction and mitigate Out-Of-Vocabulary concept drift.
3. **Event Rarity (Classical Baseline)**: Scores sequence windows based on the negative log frequency of constituent events derived from training distributions.
4. **Bigram Transition (Classical Baseline)**: Evaluates sequences based on bigram transition probabilities, applying a default penalty probability to unseen OOV transitions.
5. **Count Z-Score (Classical Baseline)**: Calculates anomaly scores based on standard deviations from mean event counts within window blocks.
6. **Count Z-Score (OOV-neutral)**: An optimized variant of the Count Z-Score model designed to remain robust against unseen Out-Of-Vocabulary tokens, preventing spurious anomaly flags from novel but benign events.

### Synthetic Anomaly Injection
The framework includes a synthetic testing substrate built from known clean, non-overlapping test windows. Controlled permutations are injected to measure model sensitivity to specific failure modes:
- `substitute`: Replaces tokens with random known training events.
- `rare`: Replaces tokens with rarely occurring known events.
- `oov`: Replaces tokens with unseen `<UNK>` tokens.
- `repeat`: Duplicates the preceding log event.
- `shuffle`: Permutes event sequence order.

---

## 📈 Benchmark Performance (Version 9.0 Evaluation)

All models utilize a 99th-percentile threshold calibrated on the normal validation set. 

### Primary Full BGL Test Evaluation
| Model | AUC | AP | Accuracy | Precision | Recall | F1-Score | FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Count Z-Score (OOV-neutral)**| - | - | - | - | - | **0.3574** | - |
| **Standard LSTM AE** | 0.5972 | 0.1781 | 94.34% | 1.0000 | 0.1166 | 0.2088 | 0.0000 |
| **OOV-Robust LSTM AE** | **0.6578** | 0.1260 | 92.62% | 0.1919 | 0.0477 | 0.0764 | 0.0137 |
| **Bigram Transition** | 0.5784 | 0.0776 | 93.61% | 0.6567 | 0.0036 | 0.0072 | 0.0001 |
| **Event Rarity** | 0.5674 | 0.0730 | 93.60% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Count Z-Score** | 0.5387 | 0.0684 | 93.60% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Secondary Disjoint Real Test (Non-Overlapping)
| Model | AUC | AP | Accuracy | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Standard LSTM AE** | 0.5298 | 0.3374 | 78.25% | 0.2301 |
| **OOV-Robust LSTM AE** | **0.5992** | **0.3461** | 74.90% | 0.0839 |
| **Classical Baselines** | ~0.4900 | ~0.2400 | 75.00% | 0.0000 |

*Note: The Count Z-Score (OOV-neutral) currently leads the primary F1-Score evaluation. While the OOV-Robust model improves overall discriminative power (AUC), the strict 99th-percentile threshold restricts its F1-score due to precision-recall trade-offs.*

---

## ⚙️ Training & Calibration Details

* **Environment**: Built for Google Colab with GPU acceleration (T4 GPU) and Google Drive integration for artifact persistence (`/content/drive/MyDrive/BGL Log/`).
* **Optimization**: Deep learning models are trained using the Adam optimizer with a learning rate of `0.001` and batch size of `512`.
* **Early Stopping**: Validation loss is monitored with a patience of 3 epochs.
* **Thresholding**: Decision thresholds for all models are dynamically calibrated at the **99th-percentile** of their validation set scores.
* **Metrics**: Evaluated on AUC, Average Precision (AP), Accuracy, Precision, Recall, F1-Score, False Positive Rate (FPR), and False Negative Rate (FNR).

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
git clone https://github.com/AkashKumar2292/deep-learning-log-anomaly-detection.git
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
1. Upload `notebooks/Log_Anomaly_Detection.ipynb` to Google Colab.
2. Mount your Google Drive containing `BGL.log.zip` or upload the dataset directly.
3. Enable GPU Hardware Acceleration (Runtime > Change runtime type > T4 GPU).
4. Execute cells sequentially.

## 📜 License
Distributed under the MIT License.