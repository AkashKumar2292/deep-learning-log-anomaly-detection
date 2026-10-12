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
- **Dataset Used**: [BGL (BlueGene/L Supercomputer Log)](https://github.com/logpai/loghub)
- **Full Dataset**: ~4.7 Million lines (~348,460 ground-truth anomalies)
- **Data Processing**:
  1. **Regex Parsing**: Maps unstructured log lines into structured tabular fields (`Timestamp`, `Node`, `Component`, `Level`, `Content`).
  2. **Dynamic Variable Masking**: Extracts memory addresses, IP addresses, node IDs, and numbers into generalized **Event Templates**.
  3. **Direct Integer Tokenization**: Maps template hashes to unique integer indices (`Event_Token`).
  4. **Sliding Windowing**: Groups consecutive log event tokens into sequence vectors of size 20 (step size = 5).
  5. **Sequence Ground Truth**: A window is labeled as an anomaly (`Is_Anomaly = 1`) if it contains $\ge 1$ ground-truth anomalous log.

---

## 🏗️ Architecture & Version 2.0 Methodology

### 1. Chronological Data Splitting (Leakage Prevention)
To prevent temporal data leakage across sliding log windows, the 4.7M dataset is strictly split chronologically **before** window generation:
- **Train Set (70%)**: Used exclusively to train the Autoencoder on normal sequences (`Is_Anomaly = 0`).
- **Validation Set (10%)**: Used to dynamically calibrate the anomaly reconstruction threshold.
- **Test Set (20%)**: Unseen sequence data evaluated against the frozen threshold.

### 2. LSTM Autoencoder (Deep Learning)
- **Model Layers**:
  - `Embedding Layer`: Integer event tokens $\to$ 16D dense vectors.
  - `Encoder (LSTM)`: Compresses 20-log window into a 32D hidden state vector.
  - `Decoder (LSTM)`: Reconstructs the compressed hidden state back to the 20-log sequence.
  - `Linear Output Layer`: Predicts reconstruction probabilities over the event vocabulary.
- **Loss & Thresholding**: Trained using Cross-Entropy loss over healthy log patterns for 15 epochs. Anomaly scores are evaluated per window, and sequences exceeding the validation-calibrated 90th percentile reconstruction error threshold are flagged as anomalies.

### 3. Rule-Based Severity Baseline
- Scans sequence windows for critical log severity levels (`WARNING`, `ERROR`, `SEVERE`, `FATAL`, `FAILURE`) identified during training. Flags a sequence as an anomaly if it contains $\ge 1$ critical token.

---

## 📈 Performance & Results (4.7M Log Benchmark v2.0)

Evaluated under strict chronological sequence isolation (188,536 test windows):

| Model / Approach | Precision (Anomaly) | Recall (Anomaly) | F1-Score | Overall Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Rule-Based Severity Baseline** | 0.54 | 0.24 | **0.3275** | **94%** |
| **LSTM Autoencoder (DL)** | 0.19 | 0.36 | **0.2486** | **86%** |

> **Key Takeaway**: Under strict temporal evaluation without sequence leakage, the rule-based filter maintains higher precision, while the LSTM Autoencoder demonstrates higher recall in discovering sequence-level anomalies without requiring explicit error keyword flags.

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