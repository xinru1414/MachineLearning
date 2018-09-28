from typing import List
import math
import matplotlib.pyplot as plt
import numpy as np
import sys


def lp(X: List[int], theta: float, alpha: int, beta: int) -> float:
	"""
	This function computes the log_posterior of theta
	"""
	#return math.factorial(alpha + beta - 1) * theta ** (alpha - 1) * (1 - theta) ** (beta -1) / math.factorial(alpha -1) * math.factorial (beta - 1)
	return (sum(X) + beta - 1) * math.log(1 - theta) + ((len(X)) + alpha - 1 ) * math.log(theta)

def plot_map (X: List[int], thetas: List[float], alpha: int, beta: int):
	# compute the log_likelihood
	lp_1 = [lp(X[0:1000], theta, alpha, beta) for theta in thetas]
	lp_2 = [lp(X[0:10000], theta, alpha, beta) for theta in thetas]
	lp_3 = [lp(X, theta, alpha, beta) for theta in thetas]

	plt.subplot(3,1,1)
	plt.plot(thetas, lp_1)
	plt.ylabel('log_posterior of theta')
	max_y = max(lp_1)
	min_y = min(lp_1)

	pos_x = lp_1.index(max_y)
	max_x = thetas[pos_x]

	plt.xticks(np.arange(min(thetas), max(thetas)+0.01, 0.07))
	plt.ylim(math.ceil(min_y), 0)
	plt.annotate('MAP(' + str(round(max_x,2)) + ',' + str(round(max_y,2)) + ')', xy=(max_x, max_y), arrowprops=dict(facecolor='black', shrink=0.01),)

	plt.subplot(3,1,2)
	plt.plot(thetas, lp_2)
	plt.ylabel('log_posterior of theta')
	max_y = max(lp_2)
	min_y = min(lp_2)

	pos_x = lp_2.index(max_y)
	max_x = thetas[pos_x]

	plt.xticks(np.arange(min(thetas), max(thetas)+0.01, 0.07))
	plt.annotate('MAP(' + str(round(max_x,2)) + ',' + str(round(max_y,2)) + ')', xy=(max_x, max_y), arrowprops=dict(facecolor='black', shrink=0.01),)


	plt.subplot(3,1,3)
	plt.plot(thetas, lp_3)
	plt.xlabel('theta')
	plt.ylabel('log_posterior of theta')
	max_y = max(lp_3)
	min_y = min(lp_3)

	pos_x = lp_3.index(max_y)
	max_x = thetas[pos_x]

	plt.xticks(np.arange(min(thetas), max(thetas)+0.01, 0.07))
	plt.ylim(math.ceil(min_y), 0)
	plt.annotate('MAP(' + str(round(max_x,2)) + ',' + str(round(max_y,2)) + ')', xy=(max_x, max_y), arrowprops=dict(facecolor='black', shrink=0.01),)


	plt.show()

def main():
	file_1 = sys.argv[1]

	X = open(file_1).read().splitlines()
	X = list(map(int, X))
	theta = list(np.arange(0.01, 1,0.01))

	plot_map(X, theta, 1, 2)

if __name__  == '__main__':
	main()
