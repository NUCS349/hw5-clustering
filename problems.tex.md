## Problems

## Code implementation (5 points)
Pass test cases by implementing the functions in the `code` directory. You are to 
implement KMeans, Soft K-Means, and Gaussian Mixture Models, and apply them to
various datasets in thie assignment.

Your grade for this section is defined by the autograder. If it says you got an 80/100,
you get 4 points here.

## Free response questions (2 points, 1 point per question)

1. What happens as beta in soft K-Means approaches infinity? How does this relate to 
   regular K-Means?

2. Draw a data distribution that would be better modeled using a GMM x than with soft K-Means. Explain why this is so. (Hint: think about the covariance matrix)
         
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
Since we actually DO know the labels of the handwritten digets, we can also consider the accuracy of these unsupervised approaches. For each cluster, find the most common label. Call that the label of the cluster. Find the proportion of images whose label does not match their cluster label. That's the error. Report the performance of each of these algorithms using this measure. Which algorithm performed best? Is this the same one that did best with Adjusted Mutual Information Score?

### Visualization (1 point)
Clustering can be visualized in two ways: 

1. Finding the image in the data that is closest to each mean.
2. Taking the mean of all the images that are assigned to a single cluster.

For the best performing algorithm, according to Adjusted Mutual Information
Score, do
both of these. Show us the results of the visualization.

How do the two approaches to visualizations diverge from one another? What can you 
learn from this?







