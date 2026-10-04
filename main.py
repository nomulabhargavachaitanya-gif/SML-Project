import pandas as pd
import os
from data.load_dataset import get_balanced_ag_news
from preprocessing.text_preprocessor import TextPreprocessor
from preprocessing.feature_builder import build_sparse_features
from embeddings.extract_embeddings import get_embeddings
from dimensionality_reduction.dim_reduction import reduce_dimensions, plot_variance_curve
from models.train_evaluate import train_and_test_models
from evaluation.plots import plot_tsne, plot_comparison_bar, plot_class_distribution

def main():
    try:
        OUT_DIR = "outputs"
        os.makedirs(OUT_DIR, exist_ok=True)

        # 1. Loading & EDA
        print("Step 1: Loading AG News...")
        train_df, val_df, test_df = get_balanced_ag_news(8000)
        plot_class_distribution(train_df, f"{OUT_DIR}/class_dist.png")
        
        # 2. Preprocessing
        print("Step 2: Cleaning text...")
        tp = TextPreprocessor()
        tr_clean, val_clean, ts_clean = tp.transform_series(train_df['text']), tp.transform_series(val_df['text']), tp.transform_series(test_df['text'])
        
        # 3. Sparse Features
        print("Step 3: Building sparse features...")
        # Unpacks: (tr, val, ts), (tr, val, ts), vectorizer
        bow_sets, tfidf_sets, _ = build_sparse_features(tr_clean, ts_clean, val_clean)
        tr_bow, val_bow, ts_bow = bow_sets
        tr_tfidf, val_tfidf, ts_tfidf = tfidf_sets
        
        # 4. Contextual Embeddings
        print("Step 4: Loading BERT embeddings...")
        tr_cls, tr_mean = get_embeddings(train_df['text'].tolist(), f"{OUT_DIR}/embeddings_train.pkl")
        val_cls, val_mean = get_embeddings(val_df['text'].tolist(), f"{OUT_DIR}/embeddings_val.pkl")
        ts_cls, ts_mean = get_embeddings(test_df['text'].tolist(), f"{OUT_DIR}/embeddings_test.pkl")
        
        # 5. Dim Reduction
        print("Step 5: Dimensionality reduction (PCA/SVD)...")
        tr_tfidf_red, tf_red, tf_var = reduce_dimensions(tr_tfidf, 'svd', 100)
        val_tfidf_red, ts_tfidf_red = tf_red.transform(val_tfidf), tf_red.transform(ts_tfidf)

        tr_cls_red, cls_red, cls_var = reduce_dimensions(tr_cls, 'pca', 100)
        val_cls_red, ts_cls_red = cls_red.transform(val_cls), cls_red.transform(ts_cls)

        plot_tsne(tr_cls, train_df['label'], "DistilBERT CLS", f"{OUT_DIR}/tsne_cls.png")
        plot_variance_curve(cls_var, "BERT CLS PCA", f"{OUT_DIR}/pca_cls.png")

        # 6. Training Pipeline
        experiments = {
            "BoW (Unigram)": (tr_bow, val_bow, ts_bow),
            "TF-IDF (Bigram)": (tr_tfidf, val_tfidf, ts_tfidf),
            "Reduced TF-IDF (SVD)": (tr_tfidf_red, val_tfidf_red, ts_tfidf_red),
            "DistilBERT (CLS)": (tr_cls, val_cls, ts_cls),
            "DistilBERT (Mean-Pool)": (tr_mean, val_mean, ts_mean),
            "Reduced CLS (PCA 100)": (tr_cls_red, val_cls_red, ts_cls_red)
        }

        master_results = []
        for feat, (X_tr, X_val, X_ts) in experiments.items():
            res = train_and_test_models(X_tr, train_df['label'], X_val, val_df['label'], X_ts, test_df['label'], feat)
            master_results.extend(res)

        # 7. Finalize
        results_df = pd.DataFrame(master_results)
        results_df.to_csv(f"{OUT_DIR}/final_project_metrics.csv", index=False)
        plot_comparison_bar(results_df, f"{OUT_DIR}/model_comparison.png")
        print("\nPipeline finished successfully!")

    except Exception as e:
        print(f"An error occurred during execution: {e}")

if __name__ == "__main__":
    main()