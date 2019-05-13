import numpy as np
from code import generate_cluster_data

def test_generate_cluster_data():
    n_samples = [1000, 10000]
    n_centers = [1, 2, 4, 8]
    stds = [.1, .5, 1.0, 2.0]
    n_features = [1, 2, 4]

    for n in n_samples:
        for f in n_features:
            for c in n_centers:
                for s in stds:
                    X, y = generate_cluster_data(
                        n_samples=n,
                        n_features=f,
                        n_centers=c,
                        cluster_stds=s
                    )

                    assert (X.shape == (n, f))
                    assert (y.max() == c - 1)

                    for i in range(y.max()):
                        subset = X[y == i]
                        assert (np.abs(np.std(subset) - s) < 1e-1)
