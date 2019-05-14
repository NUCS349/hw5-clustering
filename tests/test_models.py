from code import KMeans, SoftKMeans, GMM
import numpy as np
from code import load_json_data, adjusted_mutual_info
import os

datasets = [
        os.path.join('data', x)
        for x in os.listdir('data')
        if os.path.splitext(x)[-1] == '.json'
    ]

def test_kmeans():
    for data_path in datasets:
        # Load data and make sure its shape is correct
        features, targets = load_json_data(data_path)
        # make model and fit
        model = KMeans(2)
        model.fit(features)

        # predict and calculate adjusted mutual info
        labels = model.predict(features)
        acc = adjusted_mutual_info(targets, labels)
        assert (acc == 1.0)

def test_soft_kmeans():
    for data_path in datasets:
        # Load data and make sure its shape is correct
        features, targets = load_json_data(data_path)
        # make model and fit
        model = SoftKMeans(2)
        model.fit(features)

        # predict and calculate adjusted mutual info
        posteriors = model.predict(features)
        labels = np.argmax(posteriors, axis=-1)
        acc = adjusted_mutual_info(targets, labels)
        assert (acc == 1.0)

def test_gmm():
    for data_path in datasets:
        # Load data and make sure its shape is correct
        features, targets = load_json_data(data_path)
        # make model and fit
        for covariance_type in ['spherical', 'diagonal']:
            model = GMM(2, covariance_type)
            model.fit(features)

            # predict and calculate adjusted mutual info
            posteriors = model.predict(features)
            labels = np.argmax(posteriors, axis=-1)
            acc = adjusted_mutual_info(targets, labels)
            assert (acc == 1.0)
