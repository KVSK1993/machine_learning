# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 00:05:49 2017

@author: Karan Vijay Singh
"""

import numpy as np

dataset = np.genfromtxt('spambase_X.csv',delimiter=",")
resultset = np.genfromtxt('spambase_y.csv',delimiter=",")

col=dataset.shape[1] #no. of features/columns
row=dataset.shape[0] #no. of training data

data_array = np.asarray(dataset, dtype = float)
result_array=np.asarray(resultset, dtype = int)


max_pass=500
mistake=np.zeros((max_pass,1)) #horizontal
bias=1/(col+1)
eta=10/np.amax(data_array)

weight_array=(1/(col+1))*np.ones(col) # no. of features/ dimensions

'''def standardising_data(data_set):
    data_set = (data_set - np.mean(data_set, axis=0)) / np.std(data_set, axis=0)
    return data_set

data_array=standardising_data(data_array)'''

#Function to normalise data
def normalise_data(data_set):
    #data_set = (data_set - np.amin(data_set, axis=0)) / (np.amax(data_set, axis=0)-np.amin(data_set, axis=0))
    for j in range(0,data_set.shape[1]):
        maxColumn=np.amax(data_set[:,j])
        minColumn=np.amin(data_set[:,j])
        data_set[:,j]=(data_set[:,j]-minColumn)/(maxColumn-minColumn)
    return data_set    
#data_array=normalise_data(data_array)

#Winnow Implementation
for t in range(0,max_pass):
        mistake[t]=0   
        for i in range(0,row-1):
             if((result_array[i])*(np.dot(data_array[i],weight_array)+bias)<=0.0):   
                 weight_array=np.multiply(weight_array,np.exp(eta*result_array[i]*(data_array[i])))
                 bias=bias*np.exp(eta*result_array[i])
                 s=bias+np.sum(weight_array)
                 weight_array=weight_array/s
                 bias=bias/s
                 mistake[t]+=1
        #print(t,mistake[t])                 
import matplotlib.pyplot as plt
plt.plot(range(0,len(mistake)),mistake)
plt.xlabel("Number of Passes")
plt.ylabel("Number of Mistakes")
plt.title("Winnow with eta %f"%(eta))
plt.show()
