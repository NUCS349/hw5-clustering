import numpy as np

class GMM():
    def __init__(self, n_clusters, covariance_type):
        """
        This class implements a Gaussian Mixture Model updated using expectation
        maximization.

        https://en.wikipedia.org/wiki/K-means_clustering

        The EM algorithm for GMMs has two steps:

        1. Update posteriors (assignments to each Gaussian)
        2. Update Gaussian parameters (means, variances, and priors for each Gaussian)

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you break these two steps apart into separate
        functions.

        Use only numpy to implement this algorithm. 

        Args:
            n_clusters (int): Number of Gaussians to cluster the given data into.
            covariance_type (str): Either 'tied', 'full', 'spherical', 'diagonal'

        """
        self.n_clusters = n_clusters
        allowed_covariance_types = ['tied', 'full', 'spherical', 'diagonal']
        if covariance_type not in allowed_covariance_types:
            raise ValueError(f'covariance_type must be in {allowed_covariance_type}')
        self.covariance_type = covariance_type
        raise NotImplementedError()

    def fit(self, features):
        """
        Fit GMM to the given data using `self.n_clusters` number of Gaussians.
        Features can have greater than 2 dimensions. 

        Args:
            features (np.ndarray): array containing inputs of size 
                (n_samples, n_features).
        Returns:
            None (saves model - means - internally)
        """
        raise NotImplementedError()

    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict membership 
        to each Gaussian.

        Args:
            features (np.ndarray): array containing inputs of size 
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted assigment to each cluster for each sample,
                of size (n_samples,n_clusters). Each row contains the soft assignment to
                each cluster.
        """
        raise NotImplementedError()