# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:20:17 2020

@author: Prthamesh
"""

import models
#import basic_agent as ba
import numpy as np

def create_X(gray_lh_arr,patch_dim):
    patch_list = []
    print(gray_lh_arr.shape)
    for r in range(0,gray_lh_arr.shape[0]-patch_dim+1):
        for c in range(0,gray_lh_arr.shape[1]-patch_dim+1):
            patch = gray_lh_arr[r:r+patch_dim,c:c+patch_dim]
            patch = patch.reshape((patch_dim**2))
            patch_list.append(patch)
            #gray_lh_patch_dict[patch] = (r,c)
    X = (np.array(patch_list))
    print(len(X))
    return X
    
def create_Y(assigned_clusters_arr,number_of_colors,patch_dim):
    m,n = (assigned_clusters_arr.shape)
    y = np.zeros((number_of_colors,(m-patch_dim+1)*(n-patch_dim+1)))
    for r in range(0,m-patch_dim+1):
        for c in range(0,n-patch_dim+1):
            row_num = assigned_clusters_arr[r+patch_dim//2,c+patch_dim//2]
            col_num = (n-patch_dim+1)*r + c
            y[row_num,col_num] = 1
    return y

def scale_features(X):
    min_arr = np.min(X,axis=0)
    max_arr = np.max(X,axis=0)
    mean_arr = np.mean(X,axis=0)
    X = (X - mean_arr)/(max_arr - min_arr)
    return X,mean_arr,max_arr,min_arr
    
def color_right_half(recolored_lh_arr,gray_lh_arr,gray_rh_arr,patch_dim,number_of_colors,centroids_arr,assigned_clusters_arr,
                     arch,alpha,noi,method,bs):  
    X = create_X(gray_lh_arr,patch_dim)
    Y = create_Y(assigned_clusters_arr,number_of_colors,patch_dim)
    X,mean_arr,max_arr,min_arr = scale_features(X)
    # recolored_rh_arr = color_right_half(X,Y)
    nn = models.NeuralNetwork()
    nn.set_architecture(arch)
    nn.set_activations(['sigm' for _ in range(len(arch)-2)]+['sigm'])
    nn.train(X,Y,alpha,noi,method)
    recolored_rh_arr = np.zeros(gray_rh_arr.shape+(3,)) # shape of recolored right half
    for r in range(0,gray_rh_arr.shape[0]-patch_dim+1):
        for c in range(0,gray_rh_arr.shape[1]-patch_dim+1):
            gray_rh_patch = (gray_rh_arr[r:r+patch_dim,c:c+patch_dim])
            patch_flat = gray_rh_patch.reshape(1,patch_dim**2)
            patch_flat = (patch_flat - mean_arr)/(max_arr-min_arr)
            y_hat = (nn.forward_prop(patch_flat))[-1]
            index = np.argmax(y_hat)
            recolored_rh_arr[r+patch_dim//2,c+patch_dim//2,:] = centroids_arr[index]
    return recolored_rh_arr
   
'''recolored_img_arr = np.zeros((rgb_img_arr.shape))
for r in range(recolored_img_arr.shape[0]):
    recolored_img_arr[r,:recolored_lh_arr.shape[1],:] = recolored_lh_arr[r,:,:]
    recolored_img_arr[r,recolored_lh_arr.shape[1]:,:] = recolored_rh_arr[r,:,:]
recolored_img_arr = np.asarray(recolored_img_arr,dtype='uint8')
Image.fromarray(recolored_img_arr).show()'''