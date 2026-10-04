import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
from evaluation.metrics import get_performance_report

def map_clusters_to_labels(true_labels, cluster_labels):
    """Optimal assignment of clusters to labels using the Hungarian Algorithm."""
    matrix = confusion_matrix(true_labels, cluster_labels)
    row_ind, col_ind = linear_sum_assignment(-matrix)
    mapping = {col_ind[k]: row_ind[k] for k in range(len(row_ind))}
    return np.array([mapping[c] for c in cluster_labels])

def train_and_test_models(X_tr, y_tr, X_val, y_val, X_ts, y_ts, feat_name):
    """Trains and evaluates models on both Validation and Test sets."""
    print(f"\n--- Running Experiment: {feat_name} ---")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": SVC(kernel='linear'),
        "RBF SVM": SVC(kernel='rbf'),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "KMeans (Baseline)": KMeans(n_clusters=4, random_state=42, n_init=10)
    }
    
    results = []
    for name, model in models.items():
        try:
            model.fit(X_tr, y_tr)
            
            # Predict for both Validation and Test
            if "KMeans" in name:
                v_preds = map_clusters_to_labels(y_val, model.predict(X_val))
                t_preds = map_clusters_to_labels(y_ts, model.predict(X_ts))
            else:
                v_preds = model.predict(X_val)
                t_preds = model.predict(X_ts)

            v_acc = accuracy_score(y_val, v_preds)
            test_metrics = get_performance_report(y_ts, t_preds, name, feat_name)
            
            results.append({
                "Feature Space": feat_name,
                "Model": name,
                "Val Acc": round(v_acc, 4),
                "Accuracy": test_metrics["Accuracy"], 
                "Macro F1": test_metrics["Macro F1"]
            })
        except Exception as e:
            print(f"Error training {name}: {e}")
            
    return results

def get_confusion_matrices(X_tr, y_tr, X_ts, y_ts):
    """
    Representative confusion matrix using Logistic Regression.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_ts)
    return confusion_matrix(y_ts, preds)