import math
from tkinter import E
import numpy as np
from linear_regression import *
from sklearn.datasets import make_regression
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 2: Apply your Linear Regression
    In this problem, use your linear regression method implemented in problem 1 to do the prediction.
    Play with parameters alpha and number of epoch to make sure your test loss is smaller than 1e-2.
    Report your parameter, your train_loss and test_loss 
    Note: please don't use any existing package for linear regression problem, use your own version.
'''

#--------------------------

n_samples = 200
X,y = make_regression(n_samples= n_samples, n_features=4, random_state=1)
y = np.asmatrix(y).T
X = np.asmatrix(X)
Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]

#########################################
## INSERT YOUR CODE HERE


def compute_test_loss(alpha,epoch,epoch_interval):
    test_loss = 1
    while (test_loss>=0.01):
        w = train(Xtrain,Ytrain,alpha,epoch)
        # print('w=',w)

        Yhat_train = compute_yhat(Xtrain,w)
        # print('Yhat_train=',Yhat_train)

        train_loss = compute_L(Yhat_train,Ytrain)
        # print('train_loss=',train_loss)

        Yhat_test = compute_yhat(Xtest,w)
        # print('Yhat_train=',Yhat_test)

        test_loss = compute_L(Yhat_test,Ytest)
        epoch += epoch_interval
        print('epoch =',epoch,'|','test_loss =',test_loss)
        

    print('==================================================================================================')
    print('alpha = ',alpha,'|','epoch =',epoch,'|','train_loss =',train_loss,'|','test_loss =',test_loss)
    print('==================================================================================================')


compute_test_loss(alpha = 0.1, epoch = 0, epoch_interval= 1)

compute_test_loss(alpha = 0.01, epoch = 0, epoch_interval= 10)

compute_test_loss(alpha = 0.001, epoch = 0, epoch_interval= 100)

#########################################

