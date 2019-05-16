from code import KMeans, GMM
import numpy as np
from code import load_json_data, adjusted_mutual_info, generate_cluster_data
import os
import random
from scipy.stats import multivariate_normal

np.random.seed(0)
random.seed(0)

def test_kmeans_on_generated():
    n_samples = [1000, 10000]
    n_centers = [2]
    stds = [.1]
    n_features = [1, 2, 4]

    for n in n_samples:
        for f in n_features:
            for c in n_centers:
                for s in stds:
                    features, targets = generate_cluster_data(
                        n_samples=n,
                        n_features=f,
                        n_centers=c,
                        cluster_stds=s
                    )
                    # make model and fit
                    model = KMeans(c)
                    model.fit(features)

                    # predict and calculate adjusted mutual info
                    labels = model.predict(features)
                    acc = adjusted_mutual_info(targets, labels)
                    assert (acc >= .9)

def test_gmm_spec():
    features, targets = generate_cluster_data(
                        n_samples=100,
                        n_features=2,
                        n_centers=2,
                        cluster_stds=.1
                    )
    gmm = GMM(2, 'spherical')
    gmm.fit(features)

    assert (hasattr(gmm, 'means'))
    assert (hasattr(gmm, 'covariances'))
    assert (hasattr(gmm, 'mixing_weights'))

def test_kmeans_spec():
    features, targets = generate_cluster_data(
                        n_samples=100,
                        n_features=2,
                        n_centers=2,
                        cluster_stds=.1
                    )
    model = KMeans(2)
    model.fit(features)

    assert (hasattr(model, 'means'))

def test_gmm_likelihood():
    features = np.random.rand(4, 2)
    means = np.random.rand(2, 2)
    covariances = np.random.rand(2, 2)
    mixing_weights = np.array([1, 1])

    gmm = GMM(means.shape[0], 'diagonal')
    gmm.means = means
    gmm.covariances = covariances
    gmm.mixing_weights = mixing_weights

    for k in range(means.shape[0]):
        scipy_prob = multivariate_normal.logpdf(
            features, means[k], covariances[k]
        )
        gmm_prob = gmm._log_likelihood(features, k)
        assert np.allclose(scipy_prob, gmm_prob)


def _test_gmm_covariance(covariance_type):
    n_samples = [1000]
    n_centers = [2]
    stds = [.1]
    n_features = [2, 4]

    for n in n_samples:
        for f in n_features:
            for c in n_centers:
                for s in stds:
                    features, targets = generate_cluster_data(
                        n_samples=n,
                        n_features=f,
                        n_centers=c,
                        cluster_stds=s
                    )
                    # make model and fit
                    model = GMM(c, covariance_type=covariance_type)
                    print()
                    model.fit(features)
                    covariances = model.covariances
                    for cov in covariances:
                        assert (np.abs(np.sqrt(cov) - s).mean() < 1e-1)

                    # predict and calculate adjusted mutual info
                    labels = model.predict(features)
                    acc = adjusted_mutual_info(targets, labels)
                    assert (acc >= .9)


def test_gmm_spherical_on_generated():
    _test_gmm_covariance('spherical')

def test_gmm_diagonal_on_generated():
    _test_gmm_covariance('diagonal')
