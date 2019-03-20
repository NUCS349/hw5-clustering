## Problems

## Code implementation (5 points)
Pass test cases by implementing the functions in the `code` directory. You are to 
implement KMeans, Soft K-Means, and Gaussian Mixture Models, and apply them to
various datasets in thie assignment.

Your grade for this section is defined by the autograder. If it says you got an 80/100,
you get 4 points here.

## Free response questions (3 points)

1. Explain the relationship between Soft KMeans and Gaussian Mixture Models. Show the 
   relationship using the math of Soft KMeans vs the math of GMMs. What types of data can
   GMMs model that KMeans and Soft KMeans cannot?
2. What happens as beta in soft KMeans approaches infinity? How does this relate to 
   regular KMeans?
3. What distance measure do KMeans, Soft KMeans, and GMMs implicitly use for clustering?
   What limitations does this place on the types of data these algorithms can cluster?
4. Examine and plot the data contained in `data/circles.json`. Attempt to cluster this data
   using KMeans, Soft KMeans, and GMM. Plot and show the results of each clustering algorithm.
   Explain the performance of each algorithm on this data. If it did not work, explain why. 
   How could the data be transformed such that these approaches can cluster the data?

## Clustering handwritten digits (2 points)

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

### Comparing approaches (1 point)
Report the performance of each of these algorithms using the Adjusted Mutual Information
Score (implemented in `code/metrics.py` for you). Which algorithm performed best?

### Visualization (1 point)
Clustering can be visualized in two ways: 

1. Finding the image in the data that is closest to each mean.
2. Taking the mean of all the images that are assigned to a single cluster.

For the best performing algorithm in the previous section (comparing approaches), do
both of these. Show us the results of the visualization. 
How do the two approaches to visualizations diverge from one another? What can you 
learn from this?





