# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 21:34:04 2019

@author: xd3y15
"""
import os
import numpy
def shuffle_in_unison(a, b):
    a=numpy.array(a)
    b=numpy.array(b)
    assert len(a) == len(b)
    shuffled_a = numpy.empty(a.shape, dtype=a.dtype)
    shuffled_b = numpy.empty(b.shape, dtype=b.dtype)
    permutation = numpy.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return [str(item) for item in shuffled_a], list(shuffled_b)
def load_hyperparameters(Cascade=False):                      #All the parameters needed for training and setting the networks
    parameters=dict()
    parameters['Layer Hierachy']=False                        #The model of training network, False for normal cascade learning, True for hierarchical cascade learning
    if parameters['Layer Hierachy']:
       parameters['Semantic']=True                            #True for semantic, false for random
    parameters['Training epochs']=60                          #The total training epochs 
    parameters['Batch size train']=128                        #The batch size of train dataset
    parameters['Batch size test']=100                         #The batch size of test dataset
    parameters['Learning rate']=[0.01]                        #The initial learning rate of SGD
    parameters['Momentum']=0.9                                #The momentum of SGD
    parameters['Interval']=39                                 #The interval of saveing results cross all batches
    parameters['Random seed']=1                               #The seed of random initialization of weights
    parameters['Network index']=1                             #The name of network being convenient to later check, must be int
    parameters['Dropout percent']=0.0                         #The percent of dropout layer after first fully-connnected layer, from 0 to 1, 0 means no dropout 
    parameters['Description']='normal cascade learning, %.2f dropout'%parameters['Dropout percent']       #Brief description of the network
    if Cascade:
        parameters['Auxiliary size']=2                                    #The number of conv layers in the auxiliary network, corresponding to k in the manuscript
        parameters['Auxiliary network nodes number']=32                   #The number of kernals in each conv layer in the auxiliary network
        parameters['Cascade epoch interval']=10                           #The interval of updating learning rate cross all epochs
        parameters['Auxiliary linear network nodes number']=[84]          #The size of fully-connected layer in the auxiliary network 
        parameters['Cascade kernal size']=3                               #The kernal size of each conv layers
        parameters['Cascade structures']=[256,256,256,256,256,256,256,256,128,128,128,128]           #The number of kernals of each layer, the length of the vector is the total number of layers in the network 
        parameters['Cascade number of layers']=len(parameters['Cascade structures'])
        # ============================Adding the number of training epochs with respect to the depth of the network=========================================
        parameters['Layer wise adding training epoch']=False        #Ture for increasing, False for keeping same number of training epochs for each layer
        if parameters['Layer wise adding training epoch']:
            parameters['Epoch adding percent']=0.2                  #The percent of increments  
        # ========================================Construct the shape of each conv layers===========================================================================================
        for i in range(parameters['Cascade number of layers']):
            parameters['Cascade layer %d shape'%i]=[]
            if i==0:
                parameters['Cascade layer %d shape'%i]=[3,parameters['Cascade structures'][i]]
            else:
                parameters['Cascade layer %d shape'%i]=[parameters['Cascade layer %d shape'%(i-1)][1],parameters['Cascade structures'][i]] 
        parameters_path='Cascade parameters/Network %d/'%parameters['Network index']
    else:
        print('Sorry, the current version only supports cascade learning.')
    # +++++++++++++++++++++++++++++++Save all the setting as a text file++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if not os.path.exists(parameters_path):
        os.makedirs(parameters_path)
    with open (parameters_path+'parameter setting.txt','w') as text_file:
        for i,value in parameters.items():
                print((i,':',value),file=text_file)
    return parameters
   