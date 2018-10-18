import numpy as np
import scipy as sp
from scipy import io
import svm

def pred(w, X):
	yhat = np.inner(w, X)
	if yhat >= 0:
		return 1
	else:
		return -1

# computes the accuracy on a test dataset
def accuracy(X, y, w):
	n, p = np.shape(X)
	n_mistakes = 0
	for j in range(n):
		yhat = pred(w, X[j,:])
		if (y[j] > 0 and yhat < 0) or (y[j] < 0 and yhat >= 0):
			n_mistakes = n_mistakes + 1.0
	return 1 - n_mistakes/n

def main():
	# Get training and testing data
	data = sp.io.loadmat('../data/mnist2.mat')
	# training
	n, p = np.shape(data['xtrain'])

	w0 = np.zeros(p)
	T = 200
	l = 0.1

	for t in range(1,T+1):
		w = svm.train(w0, data['xtrain'], data['ytrain'], t*n, l)
		print('Train Accuracy: {0}, Test Accuracy: {1}'.format(accuracy(data['xtrain'], data['ytrain'], w),accuracy(data['xtest'], data['ytest'], w)))

if __name__ == '__main__':
	main()
