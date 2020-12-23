# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 18:43:37 2020

@author: Prathamesh
"""

import activations as activ
import numpy as np

class NeuralNetwork:
    """This is a neural network model that can be used for regression and classification tasks.
    """
    def __init__(self,w_list=None,b_list=None):
        self.w_list = w_list
        self.b_list = b_list
    
    def set_architecture(self,l):
        self.arch = l
        
    def set_activations(self,l):
        self.activ = l
    
    def initialize_parameters(self):
        w_list = []
        b_list = []
        for i in range(len(self.arch)-1):
            w_i = np.random.randn(self.arch[i+1],self.arch[i])
            w_list.append(w_i)
            b_i = np.ones((self.arch[i+1],1))
            b_list.append(b_i)
        self.w_list = w_list
        self.b_list = b_list
    
    def forward_prop(self,X):
        #print('X.shape',X.shape)
        A_list = []
        m,n = X.shape
        A = X.T
        A_list.append(A)
        for i in range(len(self.w_list)):
            Z = (np.dot(self.w_list[i],A)) + self.b_list[i]
            if self.activ[i] == 'sigm':
                A = activ.sigmoid(Z)
            elif self.activ[i] == 'relu':
                A = activ.relu(Z)
            A_list.append(A)
        return A_list
    
    def compute_cost(self,A,Y):
        return -(np.sum(Y*np.log(A)))/Y.shape[1]
        
    def train(self,X,Y,alpha,noi,method='sgd',bs=1):
        m,n = X.shape # number of examples,number of features
        self.initialize_parameters() # initialize parameters w and b before starting the iterative learning algorithm
        it = 0
        # prev_cost = 1000
        print(noi)
        for _ in range(noi): # iterate
            choices = np.array([np.random.randint(0,m) for _ in range(bs)])
            w_list_new = []
            b_list_new = []
            if method == 'gd':
                A_list = self.forward_prop(X)
                cost = self.compute_cost(A_list[-1],Y)
            elif method == 'sgd' or method == 'mbgd':
                A_list = self.forward_prop(X[choices,:].reshape(bs,n))
                cost = self.compute_cost(A_list[-1],(Y[:,choices]).reshape(self.arch[-1],bs))
            print(cost,it)
            if cost < 0.0001:
                break
            
            it += 1
            if method == 'gd':
                delta_last = A_list[-1] - Y
            elif method == 'sgd' or method == 'mbgd':
                delta_last = A_list[-1] - (Y[:,choices]).reshape(self.arch[-1],bs)
            #print('delta_last',delta_last.shape)
            dJ_dw_last = (np.dot(delta_last,(A_list[-2]).T))/m
            dJ_db_last = np.sum(delta_last,axis=1,keepdims=True)
            #print('dj_db_last',dJ_db_last.shape)
            w_last_new = self.w_list[-1] - alpha*dJ_dw_last
            b_last_new = self.b_list[-1] - alpha*dJ_db_last
            w_list_new.insert(0,w_last_new)
            b_list_new.insert(0,b_last_new)
            delta_prev = delta_last
            for i in range(len(self.w_list)-2,-1,-1):
                w = self.w_list[i+1]
                #print('w',w.shape)
                A = A_list[i+1]
                if self.activ[i] == 'sigm':
                    delta_i = (np.dot(w.T,delta_prev))*(A*(1-A))
                #elif self.activ[i] == 'tanh':
                #    delta_i = None
                elif self.activ[i] == 'relu':
                    dg_dz = (A>=1).astype('uint8')
                    delta_i = (np.dot(w.T,delta_prev))*dg_dz
                dJ_dw_i = np.dot(delta_i,(A_list[i]).T)
                dJ_db_i = np.sum(delta_i,axis=1,keepdims=True)
                w_new = self.w_list[i] - alpha*dJ_dw_i
                b_new = self.b_list[i] - alpha*dJ_db_i
                w_list_new.insert(0,w_new)
                b_list_new.insert(0,b_new)
                delta_prev = delta_i
                # prev_cost = cost
            self.w_list = w_list_new
            self.b_list = b_list_new








