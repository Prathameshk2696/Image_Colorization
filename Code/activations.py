# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 12:50:47 2020

@author: Prathamesh
"""

import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def tanh(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

def relu(z):
    bool_arr = (z>0)
    return bool_arr.astype('uint8')

