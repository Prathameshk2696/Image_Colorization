# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:58:37 2020

@author: Prthamesh
"""
# standard library module
import sys # importing sys module
sys.path.append('F:\Fall 2020\Introduction to AI\Assignments\ImageColoring\Code')

# Third-party extensions
from PIL import Image, ImageOps # Image is used to load the image, ImageOps is used to gray-scale the colored image
import numpy as np # ndarray to store image matrix
import pandas as pd

# User-defined modules
import clustering as cl # k-means clustering algorithm
import basic_agent as ba # recoloring of gray-scaled right half by basic agent
import improved_agent as ia # recoloring of gray-scaled right half by improved agent
import test_metrics as tm
from tabulate import tabulate

# load RGB image from a given path
def get_rgb_image(img_path): 
    return Image.open(img_path).convert('RGB') # return the RGB image

# get gray-scaled image of the given image
def get_gray_image(rgb_image):
    return ImageOps.grayscale(rgb_image) # return gray-scaled image

# get left half of a given rgb image
def get_left_half_rgb_img(rgb_img_arr):
    no_of_columns = (rgb_img_arr.shape)[1] # width of the image to divide it into two parts
    return rgb_img_arr[:,:(no_of_columns//2),:] # all rows, 0 : half_width , all 3 RGB channels

# get left and right halves of a given gray image
def get_two_halfs_gray_img(gray_img_arr):
    no_of_columns = (gray_img_arr.shape)[1]  # width of the image to divide it into two parts
    return gray_img_arr[:,:(no_of_columns//2)],gray_img_arr[:,(no_of_columns//2):]

# recolor left half of RGB image by k-means clustering
def recolor_left_half(rgb_lh_arr,number_of_colors,useElbowMethod=False,low=2,high=10,num_of_iters=1000):
    rgb_lh_arr_flat_shape = (rgb_lh_arr.shape[0]*rgb_lh_arr.shape[1],3)
    rgb_lh_arr_flat = rgb_lh_arr.reshape(rgb_lh_arr_flat_shape) # matrix flattened into a vector of data points (r,g,b)
    if not useElbowMethod: # if elbow method is False
        number_of_clusters = number_of_colors # k = given number of colors 
    else:
        elbow_data = cl.elbow(rgb_lh_arr_flat,low,high,num_of_iters) # compute sse for k = low to k = high
        df = pd.DataFrame(elbow_data,columns=['k','cost'])
        cl.plot_elbow_data(df)
        number_of_clusters = int(input('Enter value of k at elbow point in the plot'))
        # number_of_clusters = number_of_colors
    centroids_arr,assigned_clusters_arr_flat = cl.kmeans(rgb_lh_arr_flat,number_of_clusters,1000) # execute k-means algorithm
    assigned_clusters_arr = assigned_clusters_arr_flat.reshape((rgb_lh_arr.shape[0],rgb_lh_arr.shape[1])) # reshape into a matrix
    recolored_lh_arr = np.zeros((rgb_lh_arr.shape)) # array of zeros to recolor left-half of an image
    for index,centroid in enumerate(centroids_arr):
        bool_indices = (assigned_clusters_arr==index) # boolean array
        recolored_lh_arr[bool_indices] = centroid # vectorized assignment
    recolored_lh_arr = np.asarray(recolored_lh_arr,dtype='uint8') # convert all float numbers into 8-bit integers (0-255)
    centroids_arr = centroids_arr.astype(dtype='uint8') # convert all float numbers into 8-bit integers (0-255)
    return centroids_arr,assigned_clusters_arr,recolored_lh_arr,number_of_clusters

# recolor right half of gray-scaled image
def recolor_right_half(recolored_lh_arr,gray_lh_arr,gray_rh_arr,agent_name,**d):
    if agent_name == 'basic': # if basic AI agent
        patch_dim = d['patch_dim']
        number_of_patches = d['number_of_patches']
        recolored_rh_arr = ba.color_right_half(recolored_lh_arr,gray_lh_arr,gray_rh_arr,patch_dim,number_of_patches)
    elif agent_name == 'improved': # if improved AI agent
        patch_dim = d['patch_dim']
        number_of_colors = d['number_of_colors']
        centroids_arr = d['centroids_arr']
        assigned_clusters_arr = d['assigned_clusters_arr']
        arch = d['arch']
        alpha = d['alpha']
        noi = d['noi']
        method = d['method']
        bs = d['bs']
        recolored_rh_arr = ia.color_right_half(recolored_lh_arr,gray_lh_arr,gray_rh_arr,patch_dim,number_of_colors,
                                               centroids_arr,assigned_clusters_arr,arch,alpha,noi,method,bs)
    return recolored_rh_arr # return recolored right half of gray-scaled image

# merge left and right halfs to get the full image
def merge_two_halves(lh_arr,rh_arr):
    recolored_img_arr = np.zeros((lh_arr.shape[0],lh_arr.shape[1]+rh_arr.shape[1],3)) # array of zeros to store the full image
    for r in range(recolored_img_arr.shape[0]):
        recolored_img_arr[r,:lh_arr.shape[1],:] = lh_arr[r,:,:]
        recolored_img_arr[r,lh_arr.shape[1]:,:] = rh_arr[r,:,:]
    recolored_img_arr = np.asarray(recolored_img_arr,dtype='uint8') # convert all float numbers into 8-bit integers (0-255)
    return recolored_img_arr # return full colored image
    
rgb_img = get_rgb_image(r'C:\Users\Prthamesh\Downloads\tiger (2).jpg') # load RGB image
rgb_img_arr = np.array(rgb_img) # convert colored image into 3-dimensional ndarray
rgb_lh_arr = get_left_half_rgb_img(rgb_img_arr) # get left half of colored image
useElbowMethod = True
if not useElbowMethod:
    number_of_colors = 5 # number of colors to use for recoloring left half
else:
    number_of_colors = None # it will be determined by elbow method
# By using k-means clustering algorithm, recolor the left half of the image
centroids_arr,assigned_clusters_arr,recolored_lh_arr,number_of_colors = recolor_left_half(rgb_lh_arr,number_of_colors,useElbowMethod,low=2,high=15,num_of_iters=1000)
Image.fromarray(recolored_lh_arr).show() # show the recolored left half

gray_img = get_gray_image(rgb_img) # get gray-scaled image of the RGB image
gray_img_arr = np.array(gray_img) # convert image into ndarray
gray_lh_arr,gray_rh_arr = get_two_halfs_gray_img(gray_img_arr) # get left and right halves of the gray-scaled image.
Image.fromarray(gray_lh_arr).save(r'C:\Users\Prthamesh\Pictures\Screenshots\ImageColoringResults\Shallow_large\gray_lh_arr.png')
Image.fromarray(gray_rh_arr).save(r'C:\Users\Prthamesh\Pictures\Screenshots\ImageColoringResults\Shallow_large\gray_rh_arr.png')
Image.fromarray(gray_rh_arr).show()

# basic agent specification
basic_agent_spec = {
        'patch_dim':3, # patch dimension 
        'number_of_patches':6 # number of patches to use in k-nearest neighbors algorithm
    }

# improved agent specification
improved_agent_spec = {
        'patch_dim':5, # patch dimension
        'number_of_colors':number_of_colors, # number of clusters in k-means clustering
        'centroids_arr':centroids_arr, # array of centroids found by k-means clustering
        'assigned_clusters_arr':assigned_clusters_arr, # matrix of assigned clusters for every pixel in left half
        'arch':[25,30,30,35,30,number_of_colors], # architecture of neural network from input layer to output layer. All numbers exclude the bias unit.
        'alpha':0.000005, # learning rate
        'noi':10000, # number of iterations
        'method':'gd', # optimization method. Possible values: gd, mbgd, sgd
        'bs':10000, # batch size if mini-batch gradient descent
    }


# get the right half recolored by the basic agent
recolored_lh_arr_basic = recolor_right_half(recolored_lh_arr,gray_lh_arr,gray_lh_arr,'basic',**basic_agent_spec)
recolored_lh_arr_basic = np.asarray(recolored_lh_arr_basic,dtype='uint8')
recolored_rh_arr_basic = recolor_right_half(recolored_lh_arr,gray_lh_arr,gray_rh_arr,'basic',**basic_agent_spec)
recolored_rh_arr_basic = np.asarray(recolored_rh_arr_basic,dtype='uint8')

# get the right half recolored by improved agent
recolored_lh_arr_improved = recolor_right_half(recolored_lh_arr,gray_lh_arr,gray_lh_arr,'improved',**improved_agent_spec)
recolored_lh_arr_improved = np.asarray(recolored_lh_arr_improved,dtype='uint8')
recolored_rh_arr_improved = recolor_right_half(recolored_lh_arr,gray_lh_arr,gray_rh_arr,'improved',**improved_agent_spec)
recolored_rh_arr_improved = np.asarray(recolored_rh_arr_improved,dtype='uint8')

# merge the left and right halves for basic agent
recolored_img_arr_basic = merge_two_halves(recolored_lh_arr,recolored_rh_arr_basic)
Image.fromarray(recolored_img_arr_basic).show() # show the full image

# merge the left and right halves for improved agent
recolored_img_arr_improved = merge_two_halves(rgb_lh_arr,recolored_lh_arr_improved)
Image.fromarray(recolored_img_arr_improved).show() # show the full image

#arr_temp = np.asarray(recolored_rh_arr_improved,dtype='uint8')
#Image.fromarray(rgb_rh_arr).save(r'C:\Users\Prthamesh\Pictures\Screenshots\ImageColoringResults\Shallow_large\rgb_rh_arr.png')

no_of_columns = (rgb_img_arr.shape)[1] # width of the image to divide it into two parts
rgb_rh_arr = rgb_img_arr[:,(no_of_columns//2):,:] # 

# basic agent
mse1 = tm.mse(rgb_lh_arr,recolored_lh_arr_basic)
print('MSE :',mse1)
rmse1 = tm.rmse(rgb_lh_arr,recolored_lh_arr_basic)
print('RMSE :',rmse1)
mad1 = tm.mad(rgb_lh_arr,recolored_lh_arr_basic)
print('MAD :',mad1)
l1,cm_arr1,prec1,rec1,f1_score1,supp1 = tm.classification_metrics(recolored_lh_arr,recolored_lh_arr_basic)
print('Classes :',l1)
print('Confusion Matrix')
print(tabulate(cm_arr1,tablefmt='grid'))
print('Precision :',prec1)
print('Recall :',rec1)
print('F1-score :',f1_score1)
print('Support :',supp1)

# improved agent
mse2 = tm.mse(recolored_lh_arr,recolored_lh_arr_improved)
print('MSE :',mse2)
rmse2 = tm.rmse(recolored_lh_arr,recolored_lh_arr_improved)
print('RMSE :',rmse2)
mad2 = tm.mad(recolored_lh_arr,recolored_lh_arr_improved)
print('MAD :',mad2)
l2,cm_arr2,prec2,rec2,f1_score2,supp2 = tm.classification_metrics(recolored_lh_arr,recolored_lh_arr_improved)
print('Classes :',l2)
print('Confusion Matrix :\n',cm_arr2)
print('Precision :',prec2)
print('Recall :',rec2)
print('F1-score :',f1_score2)
print('Support :',supp2)

