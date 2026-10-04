from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def build_sparse_features(train_clean, test_clean, val_clean):
    try:
        # Unigram BoW
        bow_vec = CountVectorizer(ngram_range=(1, 1), min_df=5, max_df=0.95)
        tr_bow = bow_vec.fit_transform(train_clean)
        ts_bow = bow_vec.transform(test_clean)
        val_bow = bow_vec.transform(val_clean)

        # Bigram + Unigram TF-IDF (Prevents sparsity issues)
        tfidf_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.95)
        tr_tfidf = tfidf_vec.fit_transform(train_clean)
        ts_tfidf = tfidf_vec.transform(test_clean)
        val_tfidf = tfidf_vec.transform(val_clean)

        # Returns: (Train, Val, Test) tuples for both methods
        return (tr_bow, val_bow, ts_bow), (tr_tfidf, val_tfidf, ts_tfidf), tfidf_vec
    except Exception as e:
        print(f"Feature Building Error: {e}")
        # Return empty tuples to prevent unpacking errors in main.py
        return (None, None, None), (None, None, None), None