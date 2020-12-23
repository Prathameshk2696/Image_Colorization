# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:14:00 2020

@author: Prathamesh
"""

#import sys # importing sys module
#sys.path.append('F:\Fall 2020\Introduction to AI\Assignments\ImageColoring')
import numpy as np # importing numerical python

def create_gray_lh_patch_arr(gray_lh_arr,patch_dim):
    global gray_lh_patch_arr
    #global gray_lh_patch_dict
    patch_list = []
    print(gray_lh_arr.shape)
    for r in range(0,gray_lh_arr.shape[0]-patch_dim+1):
        for c in range(0,gray_lh_arr.shape[1]-patch_dim+1):
            patch = gray_lh_arr[r:r+patch_dim,c:c+patch_dim]
            patch_list.append(patch)
            #gray_lh_patch_dict[patch] = (r,c)
    gray_lh_patch_arr = np.array(patch_list)
    print(len(gray_lh_patch_arr))
    
def get_most_similar_gray_patches(gray_lh_arr,gray_rh_patch,patch_dim,number_of_patches):
    patch_dist_arr = (np.sum(np.sum((gray_lh_patch_arr - gray_rh_patch)**2,axis=2),axis=1))**0.5
    patch_dist_arr_sorted = np.sort(patch_dist_arr)
    min_dist_list = patch_dist_arr_sorted[:number_of_patches]
    most_similar_patches_dict = {} # (r,c):dist
    for dist in min_dist_list:
        patch_dist_arr_index = np.where(patch_dist_arr==dist)[0]
        count = 0
        while True:
            r = (patch_dist_arr_index[count]//(gray_lh_arr.shape[1]-patch_dim+1))
            c = (patch_dist_arr_index[count]%(gray_lh_arr.shape[1]-patch_dim+1))
            if (r,c) not in most_similar_patches_dict:
                most_similar_patches_dict[(r,c)] = float(dist)
                break
            count += 1
    #print(most_similar_patches_dict)
    return most_similar_patches_dict

def get_rgb_color_of_patch(recolored_lh_arr,gray_rh_patch,most_similar_patches_dict,patch_dim):
    representative_colors_dict = {}
    for (r,c) in most_similar_patches_dict:
        color_tuple = tuple(recolored_lh_arr[r+patch_dim//2,c+patch_dim//2,:])
        if color_tuple not in representative_colors_dict:
            representative_colors_dict[color_tuple] = 1
        else:
            representative_colors_dict[color_tuple] += 1
    #print(representative_colors_dict)
    max_count = max(representative_colors_dict.values())
    if max_count == 1:
        most_similar_patch_dist = min(most_similar_patches_dict.values())
        mspd_r,mspd_c = [(r,c) for (r,c) in most_similar_patches_dict
                         if most_similar_patches_dict[(r,c)]==most_similar_patch_dist][0]
        representative_color = recolored_lh_arr[mspd_r+patch_dim//2,
                                                mspd_c+patch_dim//2,
                                                :]
        return representative_color
    else:
        majority_representative_color = [color_tuple for color_tuple in representative_colors_dict
                                         if representative_colors_dict[color_tuple]==max_count][0]
        return majority_representative_color
    
def color_right_half(recolored_lh_arr,gray_lh_arr,gray_rh_arr,patch_dim,number_of_patches):
    global gray_lh_patch_arr
    recolored_rh_arr = np.zeros(gray_rh_arr.shape+(3,)) # shape of recolored right half
    create_gray_lh_patch_arr(gray_lh_arr,patch_dim) # create flattened array of all the patches of gray left half
    for r in range(0,gray_rh_arr.shape[0]-patch_dim+1):
        for c in range(0,gray_rh_arr.shape[1]-patch_dim+1):
            print(r,c)
            gray_rh_patch = gray_rh_arr[r:r+patch_dim,c:c+patch_dim]
            most_similar_patches_dict = get_most_similar_gray_patches(gray_lh_arr,
                                                                  gray_rh_patch,
                                                                  patch_dim,
                                                                  number_of_patches)
            patch_color = get_rgb_color_of_patch(recolored_lh_arr,
                                                gray_rh_patch,
                                                most_similar_patches_dict,
                                                patch_dim
                                                )
            recolored_rh_arr[r+patch_dim//2,c+patch_dim//2,:] = patch_color
    recolored_rh_arr = np.asarray(recolored_rh_arr,dtype='uint8')
    del gray_lh_patch_arr
    return recolored_rh_arr


