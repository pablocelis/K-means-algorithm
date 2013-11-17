from numpy import matrix
import numpy as np
import random
import collections


#   findClosestCentroids(X, centroids) returns the index of the closest 
#   centroids for a dataset X where each row is a single example. 
#   idx = m x 1 is a vector of centroid assignments 
#   (i.e., each entry in range [0..k-1])

def findClosestCentroids(X, centroids):

    # Set k, the number of centroids 
    k = np.size(centroids,0)
    # You need to return the following variables correctly.
    idx = np.mat(np.zeros((np.size(X,0),1)), dtype=int)

    distance = np.mat(np.zeros((k,1)))
    
    for i, vect in enumerate(X):
        for p, pnt in enumerate(centroids):
            distance[p] = np.linalg.norm(vect-pnt)
        idx[i] = np.argmin(distance)
    return idx


#   computeCentroids(X, idx, k) returns the new centroids by 
#   computing the means of the data points assigned to each centroid. It is
#   given a dataset X where each row is a single data point, a vector
#   idx of centroid assignments (i.e., each entry in range [0..k-1]) for each
#   example, and k, the number of centroids. You should return a matrix of
#   centroids, where each row of centroids is the mean of the data points
#   assigned to it.

def computeCentroids(X, idx, k):

    m,n = np.shape(X)
    centroids = np.mat(np.zeros((k,n)))
 
    num_points = np.mat(np.zeros((k,1)))   # Get the number of vectors in each centroid
    
    for i, vect in enumerate(X):
        c = idx[i]              # Return the centroid assigned to this vector
        num_points[c] += 1
        centroids[c] += vect    # Sum the group of vectors of each centroid

    for i in range(k):
        centroids[i] = centroids[i]/num_points[i]

    return centroids 

#   runkMeans(X, initial_centroids, max_iters) runs the k-Means algorithm 
#   on data matrix X, where each row of X is a single example. It uses 
#   initial_centroids used as the initial centroids. max_iters specifies 
#   the total number of interactions of k-Means to execute. runkMeans returns 
#   centroids, a k x n matrix of the computed centroids and idx, a m x 1 
#   vector of centroid assignments (i.e., each entry in range [0..k-1])
#
def runkMeans(X, initial_centroids, max_iters):
    
    m,n = np.shape(X)
    k = np.size(initial_centroids,0)
    centroids = initial_centroids
    idx = np.mat(np.zeros((m,1)))

    for i in range(max_iters):
        idx = findClosestCentroids(X, centroids)
        centroids = computeCentroids(X, idx, k)
        kMeansInitCentroids(X, k)

    return [centroids, idx]


#   kMeansInitCentroids(X, k) returns k initial centroids to be
#   used with the k-Means on the dataset X
def kMeansInitCentroids(X, k):

    centroids = np.mat(np.zeros((k,np.size(X,1))))

    centroids = random.sample(X, k)

    return centroids


