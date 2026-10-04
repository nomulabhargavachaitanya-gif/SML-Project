from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plt

def reduce_dimensions(data, method='pca', n_components=100, reducer=None):
    try:
        if reducer is None: # For Training Data
            if method.lower() == 'pca':
                reducer = PCA(n_components=n_components)
            else:
                reducer = TruncatedSVD(n_components=n_components)
            reduced_data = reducer.fit_transform(data)
            return reduced_data, reducer, reducer.explained_variance_ratio_
        else: # For Val/Test Data (No Leakage!)
            reduced_data = reducer.transform(data)
            return reduced_data, reducer, None
    except Exception as e:
        print(f"Dimension Reduction Error: {e}")
        return data, None, None

def plot_variance_curve(var_ratio, title, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(var_ratio) + 1), var_ratio.cumsum(), marker='o', linestyle='--')
    plt.title(f'Explained Variance: {title}')
    plt.xlabel('Components')
    plt.ylabel('Cumulative Variance')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()