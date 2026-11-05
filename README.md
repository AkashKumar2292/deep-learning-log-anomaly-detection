# Deep Learning-Based Anomaly Detection in Cloud-Native Applications

An end-to-end implementation comparing an unsupervised **Deep Learning approach (LSTM Autoencoder)** against a **Rule-Based Severity Baseline** for detecting anomalies in cloud application system logs using the BGL dataset.

*(Note: For archival documentation regarding earlier iterations of this project, please refer to README_6.md verbatim).*

## 📌 Problem Statement

Cloud-native applications generate vast volumes of unstructured log data. Traditional threshold-based monitoring relies on static keyword matching (e.g., searching for `ERROR` or `FATAL`), which struggles to detect complex sequential anomaly patterns and frequently triggers high false-positive rates. This project implements unsupervised **LSTM Autoencoder** architectures designed to learn normal log sequence patterns and flag reconstruction errors as anomalies, benchmarking performance against specialized OOV-robust variants and classical baselines.

## 📊 Dataset & Preprocessing Pipeline (Version 11.0)

* **Dataset**: 

  $$
  BGL (BlueGene/L Supercomputer Log)
  $$

* **Scale & Class Distribution**: \~4.7 million parsed log lines with a highly imbalanced log-level distribution (92.6% normal logs vs. 7.4% anomalous logs).

* **Data Pipeline**:

  1. **Deterministic Parsing**: Custom regular expression parsing extracts structured headers and raw content. Strict deterministic ordering is enforced using timestamps and original physical line numbers as tie-breakers. The parsed dataframe is then cached as a `.parquet` file (`parsed_bgl_v2.parquet`) to drastically speed up subsequent runs.

  2. **Vocabulary Mapping**: The training vocabulary encompasses 30,050 unique event templates, including a dedicated `<UNK>` token (Index 0) reserved for unseen events.

  3. **Chronological Splitting**: Data is split chronologically prior to windowing into 70% Training, 10% Validation, and 20% Testing sets to prevent lookahead bias.

  4. **Sliding Windows**: Sequence generation uses a sliding window size of 20. Overlapping sequences utilize a step size of 5 for training, while non-overlapping strict test evaluation utilizes a step size of 20.

## 🏗️ Methodology & Models (Version 11.0)

This pipeline evaluates six distinct anomaly detection strategies, alongside an in-depth ablation study:

1. **Standard LSTM Autoencoder**: A PyTorch-based sequence autoencoder utilizing an embedding dimension of 16 and a hidden dimension of 32. Trained solely on normal log windows.

2. **OOV-Robust Denoising LSTM Autoencoder**: Shares the architecture of the standard LSTM, but is trained on normal sequences artificially corrupted with a random `<UNK>` masking probability to force contextual reconstruction and mitigate Out-Of-Vocabulary concept drift.

3. **Event Rarity (Classical Baseline)**: Scores sequence windows based on the negative log frequency of constituent events derived from training distributions.

4. **Bigram Transition (Classical Baseline)**: Evaluates sequences based on bigram transition probabilities, applying a default penalty probability to unseen OOV transitions.

5. **Count Z-Score (Classical Baseline)**: Calculates anomaly scores based on standard deviations from mean event counts within window blocks.

6. **Count Z-Score (OOV-neutral)**: An optimized variant designed to remain robust against unseen Out-Of-Vocabulary tokens, preventing spurious anomaly flags from novel but benign events.

### OOV Mask-Probability Ablation (New in v11.0)
Version 11.0 introduces a rigorous ablation study testing the OOV-Robust model's sensitivity to varying `<UNK>` masking probabilities (`0.00`, `0.05`, `0.10`, `0.15`, and `0.20`) to identify the optimal regularization threshold for real-world unseen event distributions.

### Synthetic Anomaly Injection
The framework includes a synthetic testing substrate built from known clean, non-overlapping test windows. Controlled permutations are injected to measure model sensitivity to specific failure modes:
* `substitute`, `rare`, `oov`, `repeat`, and robust `shuffle` mechanisms.

## 📈 Benchmark Performance (Version 11.0 Evaluation)

All models utilize a 99th-percentile threshold calibrated on the normal validation set.

### Primary Full BGL Test Evaluation

| Model | AUC | AP | Accuracy | Precision | Recall | F1-Score | FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Count Z-Score (OOV-neutral)** | 0.5517 | 0.2836 | 94.96% | 0.9604 | 0.2196 | **0.3574** | 0.0006 |
| **Standard LSTM AE** | **0.6057** | 0.1940 | 94.45% | 0.9785 | 0.1327 | 0.2337 | 0.0002 |
| **OOV-Robust LSTM AE** | 0.5211 | 0.0825 | 93.73% | 0.8092 | 0.0233 | 0.0453 | 0.0004 |
| **Bigram Transition** | 0.5774 | 0.0765 | 93.63% | 0.5844 | 0.0037 | 0.0074 | 0.0002 |
| **Event Rarity** | 0.5684 | 0.0729 | 93.62% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Count Z-Score** | 0.5413 | 0.0686 | 93.62% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Secondary Disjoint Real Test (Non-Overlapping)

| Model | AUC | AP | Accuracy | Precision | Recall | F1-Score | FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Count Z-Score (OOV-neutral)** | **0.6051** | **0.3018** | 94.17% | 0.9764 | 0.2203 | **0.3595** | 0.0004 |
| **Standard LSTM AE** | 0.5354 | 0.1929 | 93.52% | 0.9777 | 0.1314 | 0.2317 | 0.0002 |
| **OOV-Robust LSTM AE** | 0.4353 | 0.0809 | 92.68% | 0.7738 | 0.0216 | 0.0421 | 0.0005 |
| **Bigram Transition** | 0.5015 | 0.0764 | 92.58% | 0.6471 | 0.0037 | 0.0073 | 0.0002 |
| **Event Rarity** | 0.4924 | 0.0730 | 92.57% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Count Z-Score** | 0.4609 | 0.0688 | 92.57% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

*Note: The optimized `Count Z-Score (OOV-neutral)` strongly outperforms other methods in maximizing precision while retaining recall in the strict secondary test. The OOV-Robust model's threshold behavior causes a sharp drop in F1-score due to an imbalanced precision-recall trade-off at the 99th percentile.*

## ⚙️ Training, Checkpoints & Calibration

* **Environment**: Built for Google Colab with GPU acceleration (T4 GPU) and Google Drive integration for artifact persistence.

* **Optimization**: Models are trained using the Adam optimizer (LR: `0.001`, Batch Size: `512`).

* **Checkpoints**: Weights are automatically saved to `best_ae_model.pth` and `best_ae_oov_aware_model.pth`. The pipeline now additionally caches ablation weights (e.g. `ablation_oov_p15.pth`).

* **Early Stopping**: Validation loss is monitored with a patience of 3 epochs.

* **Thresholding**: Decision thresholds for all models are dynamically calibrated at the **99th-percentile** of their validation set scores.

## 📂 Repository Structure

```text
deep-learning-log-anomaly-detection/
├── .gitignore
├── LICENSE
├── README.md
├── README_6.md                     # Legacy documentation (referenced verbatim)
├── requirements.txt
├── notebooks/
│   └── Log_Anomaly_Detection.ipynb # Complete prototype (2k) & full (4.7M) execution pipeline
```

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/AkashKumar2292/deep-learning-log-anomaly-detection.git
cd deep-learning-log-anomaly-detection
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook

```bash
jupyter notebook notebooks/Log_Anomaly_Detection.ipynb
```

*(Note: For the initial run, ensure `FORCE_REBUILD = True` is set in the notebook to parse the raw BGL.log and build the Parquet cache. On subsequent runs, set it to `False` for faster startup).*

### 4. Running on Google Colab

1. Upload `notebooks/Log_Anomaly_Detection.ipynb` to Google Colab.

2. Mount your Google Drive containing `BGL.log.zip` or upload the dataset directly.

3. Enable GPU Hardware Acceleration (Runtime > Change runtime type > T4 GPU).

4. Execute cells sequentially.

## 📜 License

Distributed under the MIT License.