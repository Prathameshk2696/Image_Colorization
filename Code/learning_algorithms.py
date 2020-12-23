# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 19:20:19 2020

@author: Prathamesh
"""

import numpy as np

def gradient_descent(X,y,model,alpha,noi):
    """find the local minimum of a differentiable cost function using full set of examples.
    alpha is the learning rate.
    noi is the number of iterations.
    X is the feature matrix of shape (m,n).
    y is the label vector of shape (m,1)."""
    for _ in range(noi): # iterate
        dJ_dw,dJ_db = model.get_gradients(X,y) # compute derivatives of J w.r.t all the parameters in w and b
        model.w -= alpha * dJ_dw # update all the parameters in w
        model.b -= alpha * dJ_db # update parameter b
        
def mini_batch_gradient_descent(X,y,model,alpha,noi,bs):
    """find the local minimum of a differentiable cost function using a batch of examples.
    alpha is the learning rate.
    noi is the number of iterations.
    X is the feature matrix of shape (m,n).
    y is the label vector of shape (m,1).
    bs is the batch size."""
    m,n = X.shape
    for _ in range(noi): # iterate
        choices = np.array([np.random.randint(0,m) for _ in range(bs)])
        dJ_dw,dJ_db = model.get_gradients(X[choices,:],y[choices,:]) # compute derivatives of J w.r.t all the parameters in w and b
        model.w -= alpha * dJ_dw # update all the parameters in w
        model.b -= alpha * dJ_db # update parameter b

def stochastic_gradient_descent(X,y,model,alpha,noi):
    """find the local minimum of a differentiable cost function using a batch of examples.
    alpha is the learning rate.
    noi is the number of iterations.
    X is the feature matrix of shape (m,n).
    y is the label vector of shape (m,1)."""
    m,n = X.shape
    for _ in range(noi): # iterate
        choice = np.random.randint(0,m)
        dJ_dw,dJ_db = model.get_gradients(X[choice,:].reshape(1,n),y[choice,:]) # compute derivatives of J w.r.t all the parameters in w and b
        model.w -= alpha * dJ_dw # update all the parameters in w
        model.b -= alpha * dJ_db # update parameter b
        
        
        
        