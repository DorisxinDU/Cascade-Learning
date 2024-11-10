# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 09:56:09 2019

@author: xd3y15
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
#===================================================================
class first_layer_cascade_Net(nn.Module):                                                                                                   #The first layer network of cascade learning                     
    def _int_(self,layer_index,hyper_parameters,layer_output_size):                                                                         #Initialize all necessary layers, e.g. Conv, fully-connected, max-pool and batch_norm 
        super(first_layer_cascade_Net, self).__init__()
        self.layer_index=layer_index
        self.hyper_parameters=hyper_parameters
        # =======================================The first layer block=============================================
        self.conv1 = nn.Conv2d(self.hyper_parameters['Cascade layer %d shape'%self.layer_index][0],self.hyper_parameters['Cascade layer %d shape'%self.layer_index][1], kernel_size=self.hyper_parameters['Cascade kernal size'],padding=self.hyper_parameters['Cascade kernal size'])
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_bn = nn.BatchNorm2d(self.hyper_parameters['Cascade layer %d shape'%self.layer_index][1])
        # ======================================Auxiliary network part=============================================  
        self.aux_network_size=self.hyper_parameters['Auxiliary network nodes number']
        self.conv2=nn.Conv2d(self.hyper_parameters['Cascade layer %d shape'%self.layer_index][1],self.aux_network_size,kernel_size=self.hyper_parameters['Cascade kernal size'],padding=self.hyper_parameters['Cascade kernal size'])
        self.pool_aux = nn.MaxPool2d(2, 2)
        self.conv2_bn = nn.BatchNorm2d(self.aux_network_size)
        # --------------------------------------Adding The number of Convolutional layers in the auxiliary network-----------------------------------------------
        if self.hyper_parameters['Auxiliary size']>1:
	        self.conv3=nn.Conv2d(self.aux_network_size,self.aux_network_size,kernel_size=self.hyper_parameters['Cascade kernal size'],padding=self.hyper_parameters['Cascade kernal size'])
	        self.pool_aux1 = nn.MaxPool2d(2, 2)
	        self.conv3_bn = nn.BatchNorm2d(self.aux_network_size)
        # -----------------------------------Calculate the size of conv layer output-----------------------------------------------
        min_s=self.hyper_parameters['Cascade kernal size']-2*self.hyper_parameters['Cascade kernal size']
        self.size_calculated=32
        for au_index in range(self.hyper_parameters['Auxiliary size']+1):
                    self.size_calculated=int(((self.size_calculated-min_s)/1+1)/2)                                                           #(W-F+2P)/S+1   W:input volume size, F: converlutional layer neuros, P: padding, S: stride   
        self.size_calculated_prelayer=int(((32-self.hyper_parameters['Cascade kernal size']+2*self.hyper_parameters['Cascade kernal size'])/1+1)/2)
        layer_output_size.append(self.size_calculated_prelayer)
        # -------------------------------------------------The fully-connected layers in the auxiliary network--------------------------------------------------------------------------
        self.fc1 = nn.Linear(self.aux_network_size*self.size_calculated*self.size_calculated, self.hyper_parameters['Auxiliary linear network nodes number'][0])                                            
        self.drop = nn.Dropout2d(hyper_parameters['Dropout percent'])
        self.fc3 = nn.Linear(self.hyper_parameters['Auxiliary linear network nodes number'][0],hyper_parameters['Number of classess'])  
    def forward(self, x):                                                                                                                    #The forward() pass defines the way we compute our output using the given layers and functions
        x = self.pool(F.relu(self.conv1(x)))
        x=self.conv1_bn(x)
        # ======================================Auxiliary network=============================================
        x = self.pool_aux(F.relu(self.conv2(x)))
        x=self.conv2_bn(x)
        if self.hyper_parameters['Auxiliary size']>1:                            
	        x = self.pool_aux1(F.relu(self.conv3(x)))
	        x=self.conv3_bn(x)
        x= x.view(-1, self.aux_network_size*self.size_calculated*self.size_calculated)                                          #Flatten the output of Conv layers
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x                        
class non_first_layer_cascade_Net(nn.Module):                                                                                                #The other layers network of cascade learning, excepting the first layer
    def _int_(self,layer_index,hyper_parameters,layer_output_size):			                                                                 #Initialize all necessary layers, e.g. Conv, fully-connected, max-pool and batch_norm 
        super(non_first_layer_cascade_Net,self).__init__()
        self.layer_index=layer_index
        self.hyper_parameters=hyper_parameters
        # =======================================The new added layer block=========================================
        self.new_conv1 = nn.Conv2d(self.hyper_parameters['Cascade layer %d shape'%self.layer_index][0],self.hyper_parameters['Cascade layer %d shape'%self.layer_index][1], kernel_size=self.hyper_parameters['Cascade kernal size'],padding=self.hyper_parameters['Cascade kernal size'])
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_bn = nn.BatchNorm2d(self.hyper_parameters['Cascade layer %d shape'%self.layer_index][1])
        # ======================================Auxiliary network part=============================================
        self.aux_network_size=self.hyper_parameters['Auxiliary network nodes number']
        self.conv2=nn.Conv2d(self.hyper_parameters['Cascade layer %d shape'%self.layer_index][1],self.aux_network_size,kernel_size=self.hyper_parameters['Cascade kernal size'],padding=self.hyper_parameters['Cascade kernal size'])
        self.pool_aux = nn.MaxPool2d(2, 2)
        self.conv2_bn = nn.BatchNorm2d(self.aux_network_size)
        # -------------------------Adding The number of Convolutional layers in the auxiliary network---------------------------------------------------
        if self.hyper_parameters['Auxiliary size']>1:
	        self.conv3=nn.Conv2d(self.aux_network_size,self.aux_network_size,kernel_size=self.hyper_parameters['Cascade kernal size'],padding=self.hyper_parameters['Cascade kernal size'])
	        self.pool_aux1 = nn.MaxPool2d(2, 2)
	        self.conv3_bn = nn.BatchNorm2d(self.aux_network_size)
        # ----------------------------Calculate the size of conv layer output--------------------------------------------------
        min_s=self.hyper_parameters['Cascade kernal size']-2*self.hyper_parameters['Cascade kernal size']
        self.size_calculated=layer_output_size[self.layer_index-1]
        for au_index in range(self.hyper_parameters['Auxiliary size']+1):
                    self.size_calculated=int(int((((self.size_calculated-min_s))/1+1)-2)/2+1)
        self.size_calculated_prelayer=int(int((((layer_output_size[self.layer_index-1]-self.hyper_parameters['Cascade kernal size']+2*self.hyper_parameters['Cascade kernal size']))/1+1)-2)/2+1)
        layer_output_size.append(self.size_calculated_prelayer)
        # -------------------------------------------------The fully-connected layers in the auxiliary network------------------
        self.fc1 = nn.Linear(self.aux_network_size*self.size_calculated*self.size_calculated, self.hyper_parameters['Auxiliary linear network nodes number'][0])
        self.drop = nn.Dropout2d(hyper_parameters['Dropout percent'])
        self.fc2 = nn.Linear(self.hyper_parameters['Auxiliary linear network nodes number'][0],self.hyper_parameters['Number of classess'])
    def forward(self,x):
        x = self.pool(F.relu(self.new_conv1(x)))
        x=self.conv1_bn(x)
        # ======================================Auxiliary network part=============================================
        x = self.pool_aux(F.relu(self.conv2(x)))
        x=self.conv2_bn(x)
        if self.hyper_parameters['Auxiliary size']>1:
	        x = self.pool_aux1(F.relu(self.conv3(x)))
	        x=self.conv3_bn(x)
        x= x.view(-1, self.aux_network_size*self.size_calculated*self.size_calculated)                                        #Flatten the output of Conv layers
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x
        
            
        