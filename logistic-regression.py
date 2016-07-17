# most of this code was modified from David Kaleko
# https://github.com/kaleko/CourseraML/blob/master/ex2/ex2.ipynb

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import expit #Vectorized sigmoid function 
from scipy import optimize


debug = False

datafile = 'data-formatted.csv'
# !head $datafile
loaded_data = np.loadtxt(datafile,delimiter=',',unpack=True) #Read in comma separated data

# Form the usual "X" matrix and "y" vector
X = np.transpose(np.array(loaded_data[:-1]))
y = np.transpose(np.array(loaded_data[-1:]))
m = y.size # number of training examples

# Insert the usual column of 1's into the "X" matrix
X = np.insert(X,0,1,axis=1)


# Divide the sample into two: ones with positive classification, one with null classification
pos = np.array([X[i] for i in range(X.shape[0]) if y[i] == 1])
neg = np.array([X[i] for i in range(X.shape[0]) if y[i] == 0])

#Check to make all entries are included
if debug == True:
    print('included everything?', len(pos)+len(neg) == X.shape[0])

def plotData():
    plt.figure(figsize=(10,6))
    plt.plot(pos[:,5],pos[:,1],'ro',label='positives')
    plt.plot(neg[:,5],neg[:,1],'yo',label='negatives', alpha=0.2, color='#000000')
    plt.xlabel('Average Low (°F)')
    plt.ylabel('Actual Low (°F) ')
    plt.legend()
    if debug == True:
        plt.show()




#Quick check of what expit is 
myx = np.arange(-10,10,.1)
if debug == True:
    plt.plot(myx,expit(myx))
    plt.title("Sigmoid function!")
    plt.grid(True)
    plt.show()
    
#Hypothesis function and cost function for logistic regression
def h(mytheta,myX): #Logistic hypothesis function
    return expit(np.dot(myX,mytheta))

#Cost function, default lambda (regularization) 0
def computeCost(mytheta,myX,myy,mylambda = 0.): 
    """
    theta_start is an n- dimensional vector of initial theta guess
    X is matrix with n- columns and m- rows
    y is a matrix with m- rows and 1 column
    Note this includes regularization, if you set mylambda to nonzero
    For the first part of the homework, the default 0. is used for mylambda
    """
    #note to self: *.shape is (rows, columns)
    term1 = np.dot(-np.array(myy).T,np.log(h(mytheta,myX)))
    term2 = np.dot((1-np.array(myy)).T,np.log(1-h(mytheta,myX)))
    regterm = (mylambda/2) * np.sum(np.dot(mytheta[1:].T,mytheta[1:])) #Skip theta0
    return float( (1./m) * ( np.sum(term1 - term2) + regterm ) )

# Check theta as zeros, and what cost returns
initial_theta = np.zeros((X.shape[1],1))
computeCost(initial_theta,X,y)

# An alternative to OCTAVE's 'fminunc' we'll use some scipy.optimize function, "fmin"
# Note "fmin" does not need to be told explicitly the derivative terms
# It only needs the cost function, and it minimizes with the "downhill simplex algorithm."
# http://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.optimize.fmin.html

def optimizeTheta(mytheta,myX,myy,mylambda=0.):
    result = optimize.fmin(computeCost, x0=mytheta, args=(myX, myy, mylambda), maxiter=400, full_output=True)
    return result[0], result[1]

theta, mincost = optimizeTheta(initial_theta,X,y)
# 'That's pretty cool. Black boxes ftw'

# Call your costFunction function using the optimal parameters of θ. 
# You should see that the cost of ~.15"
if debug == True:
    print(computeCost(theta,X,y))

# Plotting the decision boundary: two points, draw a line between
# Decision boundary occurs when h = 0, or when
# theta0 + theta1*x1 + theta2*x2 = 0
# y=mx+b is replaced by x2 = (-1/thetheta2)(theta0 + theta1*x1)

boundary_xs = np.array([np.min(X[:,1]), np.max(X[:,1])])
boundary_ys = (-1./theta[2])*(theta[0] + theta[1]*boundary_xs)

if debug == True:
    plotData()
    plt.plot(boundary_xs,boundary_ys,'b-',label='Decision Boundary')
    plt.legend()
    plt.show()

def makePrediction(mytheta, myx):
    return h(mytheta,myx) >= 0.5

# Compute the percentage of samples correct:
pos_correct = float(np.sum(makePrediction(theta,pos)))
neg_correct = float(np.sum(np.invert(makePrediction(theta,neg))))
tot = len(pos)+len(neg)
prcnt_correct = float(pos_correct + neg_correct)/tot

if debug == True:
    print("Fraction of training samples correctly predicted: %f." % prcnt_correct)


#For a student with an Exam 1 score of 45 and an Exam 2 score of 85, 
#you should expect to see an admission probability of 0.776.


# Average min
# Average max

# print(date)

# X_row_length = len(X[:, 0])
# X_column_length = len(X[0, :])


# print(X[2:1])
X_column_1 = 'Average min'
X_column_2 = 'Average max'
X_column_3 = 'mean actual'
X_column_4 = 'max temp actual'
X_column_5 = 'min temp actual'

    
      # day, column
# print(X[0, 1])
# print(X[0, 1:])

print('\n \n Pick a day out of the year (0-364)')
date = int(input(''))

average_min = X[date, 1]
average_max = X[date, 2]

print('Average min temp that day ==', average_min, 'F')
print('Average max temp that day ==', average_max, 'F')

print('\n \n Input the mean temp (F)')
mean_temp = int(input(''))

print('\n \n Input the max temp (F)')
max_temp = int(input(''))

print('\n \n Input the min temp (F)')
min_temp = int(input(''))

# print('Pick max temp (°F)')
# max_temp = input('')

# print('Pick min temp (°F)')
# min_temp = input('')
prediction = (h(theta,np.array([1, average_min, average_max, mean_temp, max_temp, min_temp])))
print('There is a', prediction * 100, '% change trump would have tweeted about climate change')

