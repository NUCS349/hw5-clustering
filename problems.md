## Problems

## Code implementation (5 points)
Pass test cases by implementing the functions in the `code` directory. You are to
implement KMeans and Gaussian Mixture Models, and apply them to
the dataset of Gaussian blobs generated in this assignment.

Your grade for this section is defined by the autograder. If it says you got an 80/100,
you get 4 points here.

Suggested order for passing test cases:
1. test_kmeans_spec
2. test_kmeans_on_generated
4. test_gmm_spec
5. test_gmm_likelihood
6. test_gmm_spherical_on_generated
7. test_gmm_diagonal_on_generated

## Free response questions (2 points, 1 point per question)

1. What happens as beta in soft K-Means approaches infinity? How does this relate to
   regular K-Means?

2. Draw a data distribution using [ml-playground](http://ml-playground.com/) that would be better modeled using a GMM x than with soft K-Means. Explain why this is so. Include the plot you made in ML playground. (Hint: think about the covariance matrix)

## Clustering handwritten digits (3 points, total)

In this problem, we will attempt to cluster handwritten digits contained in the MNIST dataset.
Because we do not use the labels themselves during training of KMeans or GMMs (i.e., we are performing _unsupervised_ learning),
we can measure our performance directly on the dataset we used to train. For that reason, we will work only with the test partition of the MNIST dataset on this portion of the assignment.

Download the test partition of the MNIST dataset from the following links:

- [images](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz)
- [labels](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz)

Use the data loader found in `code/mnist.py` to load the MNIST test images and labels.

Rather than training on the entire test partition, we will use a _class-balanced_ partition of the test set (i.e., we will use an equal number of examples from each class).
Find the digit that has the fewest examples in the MNIST test dataset. Let the number of examples of this digit in the MNIST test dataset be _n_.
Randomly sample _n_ examples from each digit in the MNIST test dataset without replacement. This will be the subset of examples that we will use in our experiments.

Now you will test your clustering algorithms on this class-balanced partition.
Each image in MNIST has dimensionality 28x28. Flatten this representation such that each image
is mapped to a 784-dimensional vector (28*28), where each element of the vector is the
intensity of the corresponding pixel in the image. Cluster these vectors into 10 clusters (i.e., `n_clusters`=10)
using the following algorithms:

- KMeans
- Gaussian Mixture Model

NOTE: IF YOUR IMPLEMENTATION OF GMM/KMEANS IS TOO SLOW FOR THESE EXPERIMENTS (OR YOUR IMPLEMENTATION 
DOESN'T WORK), YOU MAY USE THE IMPLEMENTATION CONTAINED IN SCIKIT-LEARN TO SOLVE THE FREE RESPONSE: 

https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html

### Comparing approaches without labels (1 point)
Report the performance of each of these algorithms using the Adjusted Mutual Information
Score (implemented in `code/metrics.py` for you). Which algorithm performed best?

### Comparing approaches with labels (1 point)
Since we actually _do_ know the labels of the handwritten digits, we can also consider the accuracy of these unsupervised approaches.
For each cluster, find the most common label of all examples in that cluster. Call that the label of the cluster.
Find the proportion of images whose label matches their cluster label. That's the accuracy.
Report the performance of each of these algorithms using this measure. Which algorithm performed best? Is this the same one that did best with Adjusted Mutual Information Score?

### Visualization (1 point)
Visualizing examples within each cluster can help us understand what our clustering algorithms have learned. Here are two ways that a cluster can be visualized:

1. Find the mean of all examples belonging to a cluster.
2. Find the mean of all examples belonging to a cluster, then find the nearest example to the mean (i.e., the nearest neighbor).

For the best performing algorithm according to Adjusted Mutual Information Score, perform both of these visualization techniques on all 10 clusters. Show us the results of the visualization.
Note that you will have to reshape your features to 28x28 images to see the results. Use euclidean distance to determine the nearest neighbor.

What visual differences do you notice between the images generated from each of the visualization techniques? What artifacts do you notice in first technique? What do you think is the causing these artifacts?

### Generating handwritten digits (1 point - bonus)
This section is optional but can be completed for an extra point on this assignment.

To answer this question, you can use the scikit-learn implementation of the Gaussian Mixture Model:

https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html

A Gaussian Mixture Model is a *generative* model, meaning that it can not only cluster points but can also generate new points.
Pick a digit class from the testing set of MNIST (e.g., 5) and fit a Gaussian Mixture Model to all examples from that class. Train 4 GMMs with the following values of `n_components`: 1, 4, 10, and 20 (Note: a component is another term for a cluster).
For each trained GMM, sample 5 images using the `GMM.sample` function and show them to us. How does the number of components affect the quality of the sampled images?







