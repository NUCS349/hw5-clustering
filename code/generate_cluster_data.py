import numpy as np

def generate_cluster_data(
    n_samples=100, 
    n_features=2,
    n_centers=2,
    cluster_stds=1.0
):
    """
    Generate numpy arrays that are clusterable into `n_centers` clusters using your
    implementations of KMeans, Soft KMeans, and Gaussian Mixture Model. This function
    is similar to one in scikit-learn `make_blobs`: 

    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html

    You can look at the documentation for inspiration but do not import it
    to implement this function.

    The generated data should be a set of Gaussian blobs in n_features-dimensional 
    space. The means (e.g. locations) of these blobs are generated randomly by you.

    Args:
        n_samples (int): Number of samples to generate
        n_features (int): Number of features for each sample
        n_centers (int): Number of clusters to generate
        cluster_stds (float or sequence of floats): standard deviation for each cluster.
            If a single float, then each cluster has the same standard deviation. If a 
            sequence, the length of the sequence should match n_centers, and each cluster
            will have that as the standard deviation.
    Returns:
        X (np.ndarray of shape (n_samples, n_features): A numpy array containing the
            generated data. Each row represents a point in n_features-dimensional space.
            X should be clusterable into n_centers number of clusters.
        y (np.ndarray of shape (n_samples,): A numpy array containing the cluster labels
            for the generated data. Each element tells you which cluster each data point
            came from. The actual labels can be arbitrary but points belonging to 
            different clusters should have different labels. Labels should be 0 indexed,
            with labels ranging from 0,...,(n_centers-1).

    """
    raise NotImplementedError()