import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE

def plot_tsne(features, labels, title, save_path):
    # t-SNE helps us visualize high-dimensional manifolds in 2D. 
    # It preserves local structures, meaning if points are close here, 
    # the model should ideally find them easy to group or classify.
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    reduced = tsne.fit_transform(features)
    plt.figure(figsize=(10, 7))
    
    # We look for 'clusters'. If labels are mixed like confetti, the features 
    # might not be discriminative enough for the model to learn a clear decision boundary.
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette='viridis', alpha=0.6)
    plt.title(f"t-SNE: {title}")
    plt.savefig(save_path)
    plt.close()

def plot_comparison_bar(results_df, save_path):
    """Plots Accuracy and Macro F1 side-by-side using pd.melt."""
    # Comparing Accuracy vs F1 is a theoretical 'sanity check'. 
    # High accuracy but low F1 usually suggests the model is just 
    # over-fitting to the majority class and ignoring minority cases.
    df_melted = results_df.melt(id_vars=['Feature Space', 'Model'], 
                                value_vars=['Accuracy', 'Macro F1'], 
                                var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melted, x='Feature Space', y='Score', hue='Metric')
    plt.title("Model Performance Comparison (Accuracy vs F1)")
    plt.xticks(rotation=45)
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_class_distribution(df, save_path):
    # This represents the 'Prior Probability' of our classes. 
    # Most ML algorithms assume a uniform distribution; significant 
    # skewness here explains why a model might develop a bias toward certain labels.
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='label')
    plt.title("Class Distribution")
    plt.savefig(save_path)
    plt.close()