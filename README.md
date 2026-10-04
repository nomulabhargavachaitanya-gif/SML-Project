# Statistical Machine Learning: The Data Representation Showdown

**Topic:** Sparse Features (BoW/TF-IDF) vs. Contextual Embeddings (DistilBERT)

---

## Overview
This project explores the "information bottleneck" in text classification by comparing how traditional frequency-based features perform against semantically rich transformer embeddings. Using exactly **8,000 balanced samples** (2,000 per class) from the **AG News** dataset, I evaluated whether modern embeddings can enable simple statistical models to achieve state-of-the-art results.

## The Experiment Framework
I designed this pipeline to test every possible combination of 5 models across 6 different feature representations, providing a comprehensive look at model sensitivity to data density and dimensionality.

### 1. Data Representations (Feature Spaces)
*   **BoW (Unigrams):** Standard frequency counts used as a traditional baseline.
*   **TF-IDF (Bigrams):** Frequency-inverse document frequency, utilizing (1, 2) n-grams to reduce sparsity.
*   **Reduced TF-IDF:** Sparse vectors compressed to 100 components via **Truncated SVD**.
*   **DistilBERT [CLS]:** Global sentence summary extracted from the final hidden state of a frozen transformer.
*   **DistilBERT (Mean-Pool):** Averaged hidden states for all tokens in a sequence for a more comprehensive representation.
*   **Reduced CLS:** BERT embeddings compressed to 100 components via **PCA**.

### 2. Evaluated Models
*   **Supervised:** Logistic Regression, Linear SVM, RBF SVM, and KNN ($k=5$).
*   **Unsupervised:** KMeans (Baseline) using the **Hungarian Algorithm** (linear sum assignment) for optimal cluster-to-label mapping.

---

## Project Structure
The repository is modularized to maintain a clean workflow and prevent data leakage by ensuring all reducers and vectorizers are fitted strictly on training data:

*   **`data/`**: Logic for programmatic loading and 4-class balancing of the AG News dataset.
*   **`preprocessing/`**: Text cleaning pipeline including lowercasing, alphanumeric retention, and NLTK-based stop-word removal.
*   **`embeddings/`**: Script for extracting and caching frozen DistilBERT hidden states to `.pkl` files.
*   **`dimensionality_reduction/`**: Implementations for PCA and Truncated SVD compression.
*   **`models/`**: Core training loops and the Hungarian Algorithm implementation for unsupervised evaluation.
*   **`evaluation/`**: Generation of metrics, t-SNE projections, and explained variance curves.
*   **`outputs/`**: The central storage for all generated plots, `final_project_metrics.csv`, and embedding caches.
*   **`report/`**: Final LaTeX source code providing a personal research narrative and comparative analysis.

---

## Setup & Installation
Ensure you have Python 3.8+ installed.

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download NLTK resources:**
    ```bash
    python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
    ```

3.  **Run the full pipeline:**
    ```bash
    python3 main.py
    ```

---

## Key Findings
My results demonstrate that **Mean-Pool DistilBERT embeddings** paired with an **RBF SVM** provide the highest classification accuracy (approx. **89.7%**), significantly outperforming traditional sparse methods. Furthermore, the t-SNE visualizations confirm that contextual embeddings create naturally separable semantic clusters, allowing even unsupervised KMeans to achieve over **84%** accuracy.