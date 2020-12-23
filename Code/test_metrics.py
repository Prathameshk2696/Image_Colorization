# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 20:10:20 2020

@author: Prthamesh
"""

import pandas as pd
import numpy as np

def mse(y,y_pred):
    return ((np.sum((y-y_pred)**2))/(y.shape[0]*y.shape[1]))

def rmse(y,y_pred):
    return (mse(y,y_pred))**0.5

def mad(y,y_pred):
    return (np.sum(np.abs(y-y_pred)))/(y.shape[0]*y.shape[1])

def classification_metrics(y,y_pred):
    l = []
    for r in range(y.shape[0]):
        for c in range(y.shape[1]):
            l.append(tuple(y[r,c,:]))
    l = list(set(l))
    cm_arr = [[0 for c in range(len(l))] for r in range(len(l))]
    cm = pd.DataFrame(cm_arr)
    
    for r in range(len(l)): # true 
        for c in range(len(l)): # predicted
            indices = (y==l[r])
            indices2 = (y_pred==l[c])
            indices3 = (indices[:,:,0] & indices[:,:,1] & indices[:,:,2])
            indices4 = (indices2[:,:,0] & indices2[:,:,1] & indices2[:,:,2])
            indices5 = (indices3 & indices4)
            cm[r][c] = np.sum(indices5)
    cm_arr = np.array(cm)
            
    prec = [0 for _ in range(len(l))]
    rec = [0 for _ in range(len(l))]
    supp = [0 for _ in range(len(l))]
    f1_score = [0 for _ in range(len(l))]
    
    for r in range(len(l)):
        prec[r] = round(cm_arr[r][r]/np.sum(cm_arr[:,r]),4)
    
    for r in range(len(l)):
        rec[r] = round(cm_arr[r][r]/np.sum(cm_arr[r,:]),4)
    
    for r in range(len(l)):
        supp[r] = np.sum(cm_arr[r,:])
    
    for r in range(len(l)):
        f1_score[r] = round(2*prec[r]*rec[r]/(prec[r]+rec[r]),4)
        
    return l,cm_arr,prec,rec,f1_score,supp


