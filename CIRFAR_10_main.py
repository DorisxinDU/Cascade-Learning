# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 19:59:16 2019

@author: xd3y15
"""

import torch
import torchvision
from torchsummary import summary
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import matplotlib
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
#================Load Hyper parameters===================================
from parameters import load_hyperparameters
from Build_Network import first_layer_cascade_Net,non_first_layer_cascade_Net
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
font = {'size' : 18}
matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14)
#===================================================================
torch.backends.cudnn.benchmark= True                                     #cuDNN uses nondeterministic algorithms which can be disabled
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train(epoch,layer_index,network_1,train_loader1,hyper_parameters):   #Training the network and return trained network and training accuarcy
  network_1.train()
  train_correct=0                                                        #The number of correct predection
  for batch_idx, (data, target) in enumerate(train_loader1):
        data = data.to(device)                                           #pass data and target to the GPU
        target = target.to(device)
        optimizer.zero_grad()                                            #using optimizer.zero_grad() since PyTorch by default accumulates gradients
        output=network_1(data)                                           #training model uses one batch data 
        loss = nn.CrossEntropyLoss()(output, target)                     #calculating loss of each batch, softmax is built in the loss function
        loss.backward()                                                  #backpropagation
        optimizer.step()                                                 #performs a parameter update based on the current gradient 
        # --------Recording and printing the loss following the interval--------------
        if batch_idx%hyper_parameters['Interval'] == 0 or batch_idx==len(train_loader):
            if batch_idx==len(train_loader)-1:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx)* hyper_parameters['Batch size train']+len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))
            else:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx+1) * len(data), len(train_loader.dataset),100. * (batch_idx+1) / len(train_loader), loss.item()))
            train_losses.append(loss.item())                                                                                          #record the training loss
            train_counter.append((batch_idx*hyper_parameters['Batch size train']) + ((epoch-1)*len(train_loader.dataset)))            #count the number of samples has been went through
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------
        train_pre=output.data.max(1,keepdim=True)[1]                      #get the prediction of each batch
        train_correct+=train_pre.eq(target.data.view_as(train_pre)).sum() #accumulate the number of correct prediction 
  print('Train Epoch: {}[{}/{} ({:.0f}%)]\tAccuarcy: {:.2f}'.format(epoch, (batch_idx)* hyper_parameters['Batch size train']+len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader),100. * train_correct / len(train_loader.dataset)))
  return network_1,100. * train_correct / len(train_loader.dataset) 
# =====================================================================
def test(network_1, test_loader1,hyper_parameters):            #Test the network and return test accuarcy
  network_1.eval()
  test_loss_tem=[]
  correct = 0
  with torch.no_grad():
    for test_data, test_target in test_loader1:
      test_target=test_target.to(device)
      test_data=test_data.to(device)
      test_output=network_1(test_data)
      test_loss_tem.append(nn.CrossEntropyLoss()(test_output, test_target).item())
      pred = test_output.data.max(1, keepdim=True)[1]
      correct += pred.eq(test_target.data.view_as(pred)).sum()  #accumulate the number of correct prediction         
  test_loss=np.mean(test_loss_tem)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
  return 100. * correct / len(test_loader.dataset) 
#======================================================================
for times in range(5):                                          #give 5 runs for checking the uncertainty
    hyper_parameters=load_hyperparameters(Cascade=True)         #loading parameters
    torch.manual_seed(hyper_parameters['Random seed'])          #define random seed
    #================================Cifar 10 data augumentation=================================================
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # ===========================================================================================================
    layer_output_size=[]           #used to save the size of each conv layer output    
    for layer_index in range(0,hyper_parameters['Cascade number of layers']): 
        #============================================load data================================================================================
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=hyper_parameters['Batch size train'], shuffle=True, num_workers=0)         # if you get an error when running on the terminal, please change num_workers=2
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=hyper_parameters['Batch size test'],shuffle=True, num_workers=0)
        #=====================================================================================================================================
        if hyper_parameters['Layer Hierachy']:
            print('Sorry, the current version only supports cascade learning.')
        else:                                                     #Define label of normal cascade learning
            print('The number of classes is changed to ten.')
            hyper_parameters['Number of classess']=len(np.unique(np.array(train_loader.dataset.targets)))
            classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
            #            0       1       2       3      4       5      6       7        8       9
        # ===========================================================================
        if layer_index==0:
            trained_network=network = first_layer_cascade_Net()                  #construct first layer model
            network._int_(layer_index,hyper_parameters,layer_output_size)        #initialize the constructed model
        else:
            del network                                                          #delete last layer model
            new_network=non_first_layer_cascade_Net()                            #construct the new conv layer
            new_network._int_(layer_index,hyper_parameters,layer_output_size)    #initialize the new model      
            for index,child in enumerate(trained_network.children()):            #extract the conv layer from last trained model
                if index==3*(layer_index-1) and layer_index!=1:
                    layer_need_to_extract=list(child.children())[:3]
            #----------------------constract the new model for current layer------------------------------------------------
            if layer_index==1:
                network=nn.Sequential(*list(trained_network.children())[:3*layer_index],new_network)
            else:
                network=nn.Sequential(*list(trained_network.children())[:3*(layer_index-1)],*layer_need_to_extract,new_network)
            #---------------------------------------------------------------------- 
        for index_frozen_layer, child in enumerate(network.children()):          #freeze the layer extracted from last model 
            if index_frozen_layer < (3*layer_index):    
                for param in child.parameters():
                    param.requires_grad = False    
        network.to(device)                                                       #using a GPU for training, we should have also sent the network parameters to the GPU using e.g. network.to(device).
        print('+++++++For layer ',layer_index,' The network is+++++++ \n')       #print out the generated model of current layer
        summary(network,input_size=(3,32,32))
        if hyper_parameters['Layer wise adding training epoch']:                 #if adding training epoch in layer wise fashion
            hyper_parameters['Training epochs']=int(hyper_parameters['Training epochs']*(1+hyper_parameters['Epoch adding percent']))
            print('The training epoch for layer %d is: %d'%(layer_index,hyper_parameters['Training epochs']))
        # ----------------------------------------------Seeting variables for saving results-----------------------------------------------------------------------
        train_losses = []
        train_counter = []
        test_losses = []
        test_counter = [i*len(train_loader.dataset) for i in range(hyper_parameters['Training epochs'] + 1)]              #test loss for each epoch and before training
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------
        test_acc_0_epoch=test(network,test_loader,hyper_parameters)                                                       #test performance under the random initialization    
        network_index=hyper_parameters['Network index']
        path='Cifar Cascade Model/2BNCNNAux Network %d/Times %d/layer %d/'%(network_index,times,layer_index)              #specify the path of saving results if not existing, generate it
        if not os.path.exists(path):
           os.makedirs(path)
        ir_index=0 
        start = time.time()                                                                                               #the counter used for updating the learning rate
        for epoch in range(1, hyper_parameters['Training epochs'] + 1):    
            if epoch%hyper_parameters['Cascade epoch interval']==0:
                ir_index+=1
            optimizer = optim.SGD(filter(lambda p: p.requires_grad,network.parameters()), lr=hyper_parameters['Learning rate'][0]*(0.2**ir_index),momentum=hyper_parameters['Momentum'],weight_decay=5e-4)       #weight_decay is the hyperparameters of L2 regularization
            trained_network,train_acc=train(epoch,layer_index, network,train_loader,hyper_parameters)                     #training the model and return trained network and training accuracy
            end = time.time()
            test_acc=test(trained_network,test_loader,hyper_parameters)                                                   #get test performance
            #-------------#save checkpoint of model and performce-------------------------------------------------------------------------------------------------
            if epoch%hyper_parameters['Cascade epoch interval']==0 or epoch==hyper_parameters['Training epochs']:         
                torch.save({
                        'model_state_dict': trained_network.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        },path+'model epoch %d'%epoch)
                if epoch==hyper_parameters['Training epochs']: 
                   torch.save(trained_network,path+'trained model') 
            torch.save({'epoch': epoch,
                        'train loss': train_losses,
                        'test loss':test_losses,
                        'train accuracy':train_acc,
                        'test accuracy':test_acc},path+'epoch %d'%epoch)
            #------------------------------------------------------calculating the performance of each class-----------------------------------------------------------------------------------------------
            class_correct = list(0. for i in range(hyper_parameters['Number of classess']))
            class_total = list(0. for i in range(hyper_parameters['Number of classess']))
            with torch.no_grad():
                test_b=0
                for data in test_loader:
                    images, labels = data
                    images=images.to(device)
                    labels=labels.to(device)
                    outputs = trained_network(images)
                    _, predicted = torch.max(outputs, 1)
                    c = (predicted == labels).squeeze()
                    for i in range(hyper_parameters['Batch size test']):
                        label = labels[i]
                        class_correct[label] += c[i].item()
                        class_total[label] += 1    
            results_path='Cascade Results/Network %d/Times %d/layer %d/'%(network_index,times,layer_index)
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            with open (results_path+'epoch %d acc.txt'%epoch,'w') as text_file:
                for i in range(hyper_parameters['Number of classess']):
                    if class_total[i]!=0: 
                        print(('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i])),file=text_file)
                    else:
                        print(('Accuracy of %5s : %2d %%' % (classes[i], class_correct[i])),file=text_file)
        #============================plot the training and testing loss of each layer and saving the training time of each layer==============================
        fig = plt.figure()
        plt.plot(train_counter, train_losses, color='blue')
        plt.scatter(test_counter[1::], test_losses[1::], color='red')
        plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
        plt.xticks(rotation=60)
        plt.xlabel('Number of training examples seen')
        plt.ylabel('Negative log likelihood loss')
        plt.title('Layer %d'%layer_index) 
        figure_save_path='Cifar Cascade Result plot/2BNCNNAux Network %d/Times %d/layer %d/'%(network_index,times,layer_index)
        if not os.path.exists(figure_save_path): 
            os.makedirs(figure_save_path)
        fig.savefig(figure_save_path+'across_examples.png',bbox_inches='tight' )
        time_path='Cascade Time/Network %d/Times %d/layer %d/'%(network_index,times,layer_index)
        if not os.path.exists(time_path):
            os.makedirs(time_path)
        with open (time_path+'trainin time.txt','w') as text_file:
                print(('TRAIN TIME of layer %d for one epoch:'%layer_index),file=text_file)
                print(('%.2f hours'%((end-start)/60/60)),file=text_file)
