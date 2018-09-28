from typing import List
import math
import matplotlib.pyplot as plt
import numpy as np
import sys


def ll(X: List[int], theta: float) -> float:
	"""
	This function computes the log_likelihood of theta
	"""
	return len(X) * math.log(theta) + math.log(1-theta) * sum(X)

def plot_mle (X: List[int], thetas: List[float]):
	# compute the log_likelihood
	likelihoods_1 = [ll(X[0:1000], theta) for theta in thetas]
	likelihoods_2 = [ll(X[0:10000], theta) for theta in thetas]
	likelihoods_3 = [ll(X, theta) for theta in thetas]

	plt.subplot(3,1,1)
	plt.plot(thetas, likelihoods_1)
	plt.ylabel('log_likelihood of theta')
	max_y = max(likelihoods_1)
	min_y = min(likelihoods_1)

	pos_x = likelihoods_1.index(max_y)
	max_x = thetas[pos_x]

	plt.xticks(np.arange(min(thetas), max(thetas)+0.01, 0.07))
	plt.ylim(math.ceil(min_y), 0)
	plt.annotate('MLE(' + str(round(max_x,2)) + ',' + str(round(max_y,2)) + ')', xy=(max_x, max_y), arrowprops=dict(facecolor='black', shrink=0.01),)

	plt.subplot(3,1,2)
	plt.plot(thetas, likelihoods_2)
	plt.ylabel('log_likelihood of theta')
	max_y = max(likelihoods_2)
	min_y = min(likelihoods_2)

	pos_x = likelihoods_2.index(max_y)
	max_x = thetas[pos_x]

	plt.xticks(np.arange(min(thetas), max(thetas)+0.01, 0.07))
	plt.annotate('MLE(' + str(round(max_x,2)) + ',' + str(round(max_y,2)) + ')', xy=(max_x, max_y), arrowprops=dict(facecolor='black', shrink=0.01),)


	plt.subplot(3,1,3)
	plt.plot(thetas, likelihoods_3)
	plt.xlabel('theta')
	plt.ylabel('log_likelihood of theta')
	max_y = max(likelihoods_3)
	min_y = min(likelihoods_3)

	pos_x = likelihoods_3.index(max_y)
	max_x = thetas[pos_x]

	plt.xticks(np.arange(min(thetas), max(thetas)+0.01, 0.07))
	plt.ylim(math.ceil(min_y), 0)
	plt.annotate('MLE(' + str(round(max_x,2)) + ',' + str(round(max_y,2)) + ')', xy=(max_x, max_y), arrowprops=dict(facecolor='black', shrink=0.01),)


	plt.show()

def main():
	file_1 = sys.argv[1]

	X = open(file_1).read().splitlines()
	X = list(map(int, X))
	theta = list(np.arange(0.01, 1,0.01))

	plot_mle(X, theta)

if __name__  == '__main__':
	main()
