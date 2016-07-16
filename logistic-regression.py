import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datafile = 'data-formatted.csv'
#!head $datafile

loaded_data = np.loadtxt(datafile,delimiter=',',unpack=True) #Read in comma separated data

##Form the usual "X" matrix and "y" vector
X = np.transpose(np.array(loaded_data[:-1]))
y = np.transpose(np.array(loaded_data[-1:]))
m = y.size # number of training examples

##Insert the usual column of 1's into the "X" matrix
X = np.insert(X,0,1,axis=1)

def print_test():
    print(len(pos))

#Divide the sample into two: ones with positive classification, one with null classification

pos = np.array([X[i] for i in range(X.shape[0]) if y[i] == 1])
neg = np.array([X[i] for i in range(X.shape[0]) if y[i] == 0])

#Check to make all entries are included
#print('included everything?', len(pos)+len(neg) == X.shape[0])

def plotData():
    plt.figure(figsize=(10,6))
    plt.plot(pos[:,5],pos[:,1],'ro',label='positives', s=5)
    plt.plot(neg[:,5],neg[:,1],'yo',label='negatives', alpha=0.5, color='#000000')
    plt.xlabel('Average Low (°F)')
    plt.ylabel('Actual Low (°F) ')
    plt.legend()
    plt.show()
    
plotData()









print_test()