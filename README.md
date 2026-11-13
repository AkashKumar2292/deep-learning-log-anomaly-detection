# Deep Learning-Based Anomaly Detection in Cloud-Native Applications

An end-to-end implementation comparing an unsupervised **Deep Learning approach (LSTM Autoencoder)** against a **Rule-Based Severity Baseline** for detecting anomalies in cloud application system logs using the BGL dataset.

## 📌 Problem Statement

Cloud-native applications generate vast volumes of unstructured log data. Traditional threshold-based monitoring relies on static keyword matching (e.g., searching for `ERROR` or `FATAL`), which struggles to detect complex sequential anomaly patterns and frequently triggers high false-positive rates. This project implements unsupervised **LSTM Autoencoder** architectures designed to learn normal log sequence patterns and flag reconstruction errors as anomalies, benchmarking performance against specialized OOV-robust variants, OOV-aware scoring mechanisms, and classical baselines.

## 📊 Dataset & Preprocessing Pipeline (Version 13.0)

* **Dataset**: 

  $$
  BGL (BlueGene/L Supercomputer Log)
  $$

* **Scale & Class Distribution**: 4,713,493 parsed log lines with a highly imbalanced log-level distribution (92.61% normal logs vs. 7.39% anomalous logs).

* **Data Pipeline**:

  1. **Deterministic Parsing**: Custom regular expression parsing extracts structured headers and raw content. Strict deterministic ordering is enforced using timestamps and original physical line numbers (`_source_order`) as tie-breakers, verified via SHA-256 dataset fingerprinting. The parsed dataframe is cached as a `.parquet` file (`parsed_bgl.parquet`) to drastically speed up subsequent runs.
  2. **Vocabulary Mapping**: The training vocabulary encompasses 30,050 unique event templates, including a dedicated `<UNK>` token (Index 0) reserved for unseen events.
  3. **Chronological Splitting**: Data is split chronologically prior to windowing into 70% Training (3,299,445 logs), 10% Validation (471,349 logs), and 20% Testing (942,699 logs) sets to prevent lookahead bias.
  4. **Sliding Windows**: Sequence generation uses a sliding window size of 20. Overlapping sequences utilize a step size of 5 for training (659,886 total; 601,507 normal) and primary full-test evaluation (188,536 windows), while non-overlapping strict test evaluation utilizes a step size of 20 (47,134 windows, partitioned into 6,694 clean substrate windows and 40,440 disjoint real-test windows).

## 🏗️ Methodology & Models (Version 13.0)

This pipeline evaluates six distinct anomaly detection strategies, alongside OOV-aware scoring, numerical diagnostics, and statistical ablation studies:

1. **Standard LSTM Autoencoder (with OOV-Aware Scoring)**: A PyTorch-based sequence autoencoder utilizing an embedding dimension of 16 and a hidden dimension of 32 (1,001,250 parameters), trained solely on normal log windows. In Version 13.0, inference uses **OOV-aware scoring**, computing standard cross-entropy on known tokens ($ID > 0$) while scoring `<UNK>` tokens ($ID = 0$) by the negative log probability of the most confident known-vocabulary prediction (excluding class 0).

2. **OOV-Robust Denoising LSTM Autoencoder (with OOV-Aware Scoring)**: Shares the exact architecture of the standard LSTM, but is trained on normal sequences artificially corrupted with a random `<UNK>` masking probability (`0.15`) to force contextual reconstruction and mitigate Out-Of-Vocabulary concept drift, evaluated via the same OOV-aware scoring function.

3. **Event Rarity (Classical Baseline)**: Scores sequence windows based on the negative log frequency of constituent events derived from training distributions.

4. **Bigram Transition (Classical Baseline)**: Evaluates sequences based on bigram transition probabilities, applying a default penalty probability ($1/V$) to unseen OOV transitions.

5. **Count Z-Score (Classical Baseline)**: Calculates anomaly scores based on maximum absolute standard deviations from mean event counts within window blocks.

6. **Count Z-Score (OOV-neutral)**: An optimized variant that excludes `<UNK>` (token 0) from the maximum Z-score calculation, preventing severe OOV drift from saturating the anomaly score ceiling.

### OOV Mask-Probability Ablation, Chronological Slices & Statistical Tests (Version 13.0)
* **OOV-Aware Re-Scoring & Non-Parametric Tests**: Version 13.0 re-evaluates all 15 saved ablation checkpoints across `<UNK>` masking probabilities (`0.00`, `0.05`, `0.10`, `0.15`, and `0.20`) and **three random seeds** (`42`, `43`, `44`) using the OOV-aware scoring function without retraining. Both **Kruskal-Wallis** and **Seed-Blocked Friedman** tests with **Holm step-down p-value correction** are applied across AUC, AP, F1, Precision, and Recall, confirming stable performance across masking rates ($p_{\text{adj}} \ge 0.596$).
* **OOV Prevalence & Chronological Slice Analysis**: Evaluates four-way OOV quadrants and partitions the disjoint real test set into four equal consecutive segments (`Q1`–`Q4`, 10,110 windows each), demonstrating how OOV-Robust LSTM AE maintains strong discrimination during severe early vocabulary drift (`Q1` ROC-AUC of `0.9993` vs. `0.4536` for Standard LSTM AE).
* **Count Z-Score Diagnostics**: Explicitly audits zero-variance token behavior and threshold ties at the `20000.0` ceiling.

### Synthetic Anomaly Injection
The framework includes a balanced 50/50 synthetic stress-testing substrate built from 6,694 verified OOV-free, non-overlapping normal test windows (strictly index-disjoint from the secondary real-test set). Controlled permutations at fractions `0.1`, `0.2`, and `0.4` across 3 seeds measure model sensitivity to specific failure modes:
* `substitute`, `rare`, `oov`, `repeat`, and guaranteed-mutation `shuffle` mechanisms.

## 📈 Benchmark Performance (Version 13.0 Evaluation)

All models utilize a 99th-percentile threshold calibrated on the normal validation set (`83,625` windows).

### Primary Full BGL Test Evaluation

| Model | AUC | AP | Accuracy | Precision | Recall | F1-Score | FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard LSTM AE** | **0.6739** | 0.2640 | 94.52% | 0.9544 | 0.1480 | 0.2562 | 0.0005 |
| **OOV-Robust LSTM AE** | 0.6701 | 0.2715 | 94.54% | 0.9486 | 0.1520 | 0.2620 | 0.0006 |
| **Count Z-Score (OOV-neutral)** | 0.5517 | **0.2836** | **94.96%** | **0.9604** | **0.2196** | **0.3574** | 0.0006 |
| **Bigram Transition** | 0.5774 | 0.0765 | 93.63% | 0.5844 | 0.0037 | 0.0074 | 0.0002 |
| **Event Rarity** | 0.5684 | 0.0729 | 93.62% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Count Z-Score** | 0.5413 | 0.0686 | 93.62% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Secondary Disjoint Real Test (Non-Overlapping)

| Model | AUC | AP | Accuracy | Precision | Recall | F1-Score | FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OOV-Robust LSTM AE** | **0.6346** | 0.2908 | 93.64% | 0.9717 | 0.1484 | 0.2575 | 0.0003 |
| **Standard LSTM AE** | 0.6253 | 0.2775 | 93.63% | 0.9694 | 0.1474 | 0.2559 | 0.0004 |
| **Count Z-Score (OOV-neutral)** | 0.6051 | **0.3018** | **94.17%** | **0.9764** | **0.2203** | **0.3595** | 0.0004 |
| **Bigram Transition** | 0.5015 | 0.0764 | 92.58% | 0.6471 | 0.0037 | 0.0073 | 0.0002 |
| **Event Rarity** | 0.4924 | 0.0730 | 92.57% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Count Z-Score** | 0.4609 | 0.0688 | 92.57% | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

*Note: Introducing **OOV-aware scoring** in Version 13.0 resolves the `<UNK>` cross-entropy penalty distortion observed under heavy test-set vocabulary drift (where 94.22% of normal disjoint test tokens are OOV vs. 81.39% of anomalous tokens). Consequently, **OOV-Robust LSTM AE** rises from 0.4353 to **0.6346 AUC** (leading all models in AUC on the strict disjoint real test) and **Standard LSTM AE** achieves **0.6739 AUC** on the full primary test, while **Count Z-Score (OOV-neutral)** achieves the highest F1-score (**0.3595** / **0.3574**) and Average Precision (**0.3018** / **0.2836**). In contrast, standard **Count Z-Score** saturates at its `20000.0` maximum validation score due to zero-variance `<UNK>` counts ($\mu_0 = 0, \sigma_0 = 0.001$), tying the 99th-percentile threshold and yielding zero positive predictions.*

## ⚙️ Training, Checkpoints & Calibration

* **Environment**: Built for Google Colab with GPU acceleration (Tesla T4 GPU) and Google Drive integration for artifact persistence.
* **Optimization**: Models are trained using the Adam optimizer (LR: `0.001`, Batch Size: `512`).
* **Checkpoints & Exports**: Weights are automatically saved to `best_ae_model.pth` and `best_ae_oov_aware_model.pth`, alongside multi-seed ablation weights (`v11_ablation_oov_p15_seed42.pth`). Evaluation tables and statistical reports are exported to CSV (`primary_full_bgl_results.csv`, `real_test_results.csv`, `stress_test_results.csv`, `detector_thresholds.csv`, `chronological_slice_analysis.csv`, `ablation_oov_aware_per_seed.csv`, `ablation_oov_aware_summary.csv`, `ablation_kruskal_wallis_corrected.csv`, and `ablation_friedman_corrected.csv`).
* **Early Stopping**: Validation loss is monitored with a patience of 3 epochs (`MIN_DELTA = 1e-4`, `MAX_EPOCHS = 30`).
* **Thresholding**: Decision thresholds for all models are dynamically calibrated at the **99th-percentile** of their normal validation set scores (`9.013778` for Standard LSTM AE, `8.272616` for OOV-Robust LSTM AE, `16.305421` for Event Rarity, `10.310618` for Bigram Transition, `20000.000000` for Count Z-Score, and `291.326630` for Count Z-Score OOV-neutral).

## 📂 Repository Structure

```text
deep-learning-log-anomaly-detection/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── notebooks/
│   └── Log_Anomaly_Detection.ipynb # Complete prototype (smoke) & full (4.7M) execution pipeline
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
