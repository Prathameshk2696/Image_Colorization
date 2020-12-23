# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 23:50:11 2020

@author: Prathamesh
"""

import random
import numpy as np
import matplotlib.pyplot as plt

# initialize k data points to be centroids
def initialize_centroids(train_data,k):
    return random.choices(train_data,k=k)

# update centroid to new averages
def update_centroids(train_data,clusters_list,k):
    centroids_list = [] # empty list to store new centroids
    for i in range(k): # iterate through cluster numbers. O(k)
        cluster_data = train_data[clusters_list==i] # get cluster data
        centroids_list.append(cluster_data.mean(axis=0)) # compute mean of cluster data points and append it to the list
    return np.array(centroids_list)

def get_cluster_assignment(train_data,centroids_list,k):
    distances = np.zeros(((train_data.shape)[0],k)) # matrix with data points as rows and clusters as columns
    for index,centroid in enumerate(centroids_list): # O(k)
        dist = np.linalg.norm(train_data - centroid,axis=1) # vectorized. train_data is flattened matrix and centroid is vector.
        distances[:,index] = dist # assign column to dist vector
    clusters_list = np.argmin(distances,axis=1)
    return clusters_list # compute and return cluster assignments

# function to compute and return centroids and cluster assignments
def kmeans(train_data,k,num_of_iters):
    #num_of_iters = 1000 # number of iterations in k-means algorithm
    centroids_list = initialize_centroids(train_data,k) # initial centroids
    clusters_list = get_cluster_assignment(train_data,centroids_list,k) # get cluster assignments list
    for i in range(num_of_iters): # iterate 
        prev_clusters_list = clusters_list
        centroids_list = update_centroids(train_data,clusters_list,k) # update centroids to new averages
        clusters_list = get_cluster_assignment(train_data,centroids_list,k) # 
        if (clusters_list == prev_clusters_list).all():
            print('Unchanged clusters','k = ',k)
            break
    return centroids_list,clusters_list

# This consumes time
def compute_sse(train_data,centroids_list,clusters_list):
    sse = 0
    for data_index,data_point in enumerate(train_data):
        dist = np.linalg.norm(data_point - centroids_list[clusters_list[data_index]])
        sse += (dist**2)
    return sse/len(train_data)
    
def elbow(train_data,low=1,high=10,num_of_iters=1000):
    elbow_data = []
    for k in range(low,high+1):
        centroids_list,clusters_list = kmeans(train_data,k,num_of_iters)
        sse = compute_sse(train_data,centroids_list,clusters_list)
        elbow_data.append((k,sse))
    return elbow_data

def plot_elbow_data(data):
    plt.plot(data['k'],data['cost'])
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of Squared Errors')
    plt.show()
    