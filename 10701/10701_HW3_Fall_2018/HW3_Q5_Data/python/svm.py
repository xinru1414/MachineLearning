import numpy as np
import random

# Runs the SSGD on SVM objective taking in an initial weight vector
# w0, matrix of covariates Xtrain, a vector of labels ytrain.
# 'T' is the number of passes to be made through the data.
# lbda is the regularization parameter.
# Outputs the learned weight vector w.

def train(w0, Xtrain, ytrain, T, lbda):
    n,p = np.shape(Xtrain)
    for t in range(1,T+1):
        i_t = np.random.randint(0,n)
        lr = 1/(lbda*t)
        if np.multiply(ytrain[i_t], np.inner(w0, Xtrain[i_t])) < 1:
            w0 = np.add(np.multiply((1 - lbda*lr),w0),np.multiply(lr*ytrain[i_t],Xtrain[i_t]))
        else:
            w0 = np.multiply((1 - lbda*lr),w0)

    return w0