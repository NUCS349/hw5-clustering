import numpy as np

class SoftKMeans():
    def __init__(self, n_clusters, beta=1.0):
        """
        This class implements the KMeans algorithm with SOFT assignments:

        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.

        When computing the softmax function, it will be prudent to use the `logsumexp` 
        trick to avoid numerical overflow. Here's a website that will help you
        implement it:

        https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/ 

        Use only numpy to implement this algorithm. 

        Args:
            n_clusters (int): Number of clusters to cluster the given data into.
            beta (float): The aggressiveness of the softmax when generating soft assignments.

        """
        self.n_clusters = n_clusters
        raise NotImplementedError()

    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters.
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
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels. 

        Args:
            features (np.ndarray): array containing inputs of size 
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted assigment to each cluster for each sample,
                of size (n_samples,n_clusters). Each row contains the soft assignment to
                each cluster.
        """
        raise NotImplementedError()