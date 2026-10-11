# Deep Learning-Based Anomaly Detection in Cloud-Native Applications

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end implementation comparing a **Deep Learning approach (LSTM Autoencoder)** against a **Threshold-Based Baseline Rule** for detecting anomalies in cloud application system logs (BGL Dataset).

## 📌 Problem Statement
Cloud-native applications generate vast amounts of log data. Traditional threshold-based monitoring struggles to detect subtle sequence-level anomalies and dynamic failure patterns. This project implements an unsupervised **LSTM Autoencoder** to learn normal log sequence patterns and flags reconstructive errors as anomalies, benchmarking its accuracy against standard threshold rules.

---

## 📊 Dataset Overview
* **Dataset Used**: [BGL (BlueGene/L Supercomputer Log)](https://github.com/logpai/loghub)
* **Full Dataset**: ~4.7 Million lines (~348,460 anomalies)
* **Data Processing**:
  1. Regex parsing into structured log components (`Timestamp`, `Node`, `Component`, `Level`, `Content`).
  2. Dynamic mask variable extraction (Memory addresses, IP addresses, Node IDs) to form **Event Templates**.
  3. Tokenization into unique `Event Token` integers.
  4. Sequence creation using a sliding window approach (Window size = 20 logs).

---

## 🏗️ Architecture & Approach

### 1. Preprocessing & Feature Engineering
- **Regex Parsing**: Maps un-structured log lines into tabular data.
- **Event Tokenization**: Maps template hashes to unique integer indices.
- **Sliding Windowing**: Groups consecutive log events into sequence vectors of size 20.

### 2. LSTM Autoencoder (Deep Learning)
- **Training Strategy**: Trained exclusively on **Normal (0)** log sequences to learn normal sequential behavior.
- **Model Layers**:
  - `Embedding Layer`: Integer tokens $\to$ 16D dense vectors.
  - `Encoder (LSTM)`: Compresses 20-log window into a 32D hidden state vector.
  - `Decoder (LSTM)`: Reconstructs the compressed hidden state back to the 20-log sequence.
  - `Linear Output Layer`: Predicts reconstruction probabilities over the event vocabulary.
- **Anomaly Detection**: Evaluated using reconstruction loss (Cross-Entropy). Sequences exceeding the loss threshold are flagged as anomalies.

### 3. Static Threshold Baseline
- Flags sequences containing critical log severity levels (`WARNING`, `ERROR`, `SEVERE`, `FATAL`, `FAILURE`).

---

## 📈 Performance & Results (4.7M Log Lines Benchmark)

| Model / Approach | Precision (Anomaly) | Recall (Anomaly) | F1-Score | Overall Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Threshold-Based Baseline** | 0.27 | 1.00 | **0.4311** | 77% |
| **LSTM Autoencoder (DL)** | 0.75 | 0.88 | **0.8113** | **96%** |

> **Key Finding**: The Deep Learning model drastically reduces false positives (Precision jump from 0.27 to 0.75), yielding a significantly higher **F1-Score (0.8113 vs 0.4311)** compared to static thresholding.

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/deep-learning-log-anomaly-detection.git](https://github.com/YOUR_USERNAME/deep-learning-log-anomaly-detection.git)
cd deep-learning-log-anomaly-detection