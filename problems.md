## Problems

## Code implementation (5 points)
Pass test cases by implementing the functions in the `code` directory. You are to 
implement KMeans and Gaussian Mixture Models, and apply them to
various datasets in this assignment.

Your grade for this section is defined by the autograder. If it says you got an 80/100,
you get 4 points here. 

Suggested order for passing test cases:
1. test_kmeans_spec
2. test_kmeans_on_generated
3. test_kmeans_on_files
4. test_gmm_spec
5. test_gmm_likelihood
6. test_gmm_spherical_on_generated
7. test_gmm_diagonal_on_generated
8. test_gmm_on_files

## Free response questions (2 points, 1 point per question)

1. What happens as beta in soft K-Means approaches infinity? How does this relate to 
   regular K-Means?

2. Draw a data distribution using [ml-playground(http://ml-playground.com/) that would be better modeled using a GMM x than with soft K-Means. Explain why this is so. Include the plot you made in ML playground. (Hint: think about the covariance matrix)
         
## Clustering handwritten digits (3 points, total)

For this, we will attempt to cluster handwritten digits contained in the MNIST dataset.
Download the MNIST dataset:

http://yann.lecun.com/exdb/mnist/

Download the testing set (images and labels): 

- images: http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
- labels: http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

The goal is to cluster the images based on the input representation. Each image in 
MNIST has dimensionality 28x28. We will flatten this representation such that each image
is mapped to a 784-dimensional vector (28*28), where each element of the vector is the 
intensity of the corresponding pixel in the image. Cluster these vectors into 10 clusters
using the following algorithms:

- KMeans
- Soft KMeans
- Gaussian Mixture Model

### Comparing approaches without knowing labels (1 point)
Report the performance of each of these algorithms using the Adjusted Mutual Information
Score (implemented in `code/metrics.py` for you). Which algorithm performed best?

### Comparing approaches knowing labels (1 point)
Since we actually DO know the labels of the handwritten digits, we can also consider the accuracy of these unsupervised approaches. For each cluster, find the most common label. Call that the label of the cluster. Find the proportion of images whose label matches their cluster label. That's the accuracy. Report the performance of each of these algorithms using this measure. Which algorithm performed best? Is this the same one that did best with Adjusted Mutual Information Score?

### Visualization (1 point)
Clustering can be visualized in two ways: 

1. Finding the image in the data that is closest to each mean.
2. Taking the mean of all the images that are assigned to a single cluster.

For the best performing algorithm, according to Adjusted Mutual Information
Score, do
both of these. Show us the results of the visualization.

How do the two approaches to visualizations diverge from one another? What can you 
learn from this?

### Generating handwritten digits (1 point - bonus)
This section is not necessary but can be completed for an extra point on this assignment. 

To answer this question, you can use the scikit-learn implementation of the Gaussian Mixture Model:

https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html

A Gaussian Mixture Model is a *generative* model, meaning that it can not only cluster points but can also generate new points. Pick a digit class from the testing set of MNIST (say 5) and fit a Gaussian Mixture Model to the digit with the 1, 4, 10, and 20 components. Sample 5 images from the fit GMM and show them to us. How does the number of components affect the quality of the sampled images?







