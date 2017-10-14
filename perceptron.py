# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 21:23:00 2017

@author: Karan Vijay Singh
"""


import numpy as np
dataset = np.genfromtxt('spambase_X.csv',delimiter=",")
resultset = np.genfromtxt('spambase_y.csv',delimiter=",")

predict=resultset.shape[0]

data_array = np.asarray(dataset, dtype = float)

result_array=np.asarray(resultset, dtype = int)
max_pass=500
mistake=np.zeros((max_pass,1)) #horizontal

#Function to normalise the data
def normalise_data(data_set):
    #data_set = (data_set - np.amin(data_set, axis=0)) / (np.amax(data_set, axis=0)-np.amin(data_set, axis=0))
    for j in range(0,data_set.shape[1]):
        maxColumn=np.amax(data_set[:,j])
        minColumn=np.amin(data_set[:,j])
        data_set[:,j]=(data_set[:,j]-minColumn)/(maxColumn-minColumn)
    return data_set    

#data_array=normalise_data(data_array)


col=data_array.shape[1] #no. of features/columns
row=data_array.shape[0] #no. of training data
bias=0
weight_array=np.zeros(col) # no. of features/ dimensions


#Perceptron Algorithm
for t in range(0,max_pass):
        mistake[t]=0   
        for i in range(0,row):
            
            if((result_array[i])*(np.dot(data_array[i],weight_array)+bias)<=0):
                weight_array+=result_array[i]*data_array[i]
                bias +=result_array[i]    
                mistake[t]+=1
                         
        #print(t,mistake[t])  
        
        
import matplotlib.pyplot as plt
plt.plot(range(0,len(mistake)),mistake)
plt.xlabel("Number of Passes")
plt.ylabel("Number of Mistakes")
plt.show()
