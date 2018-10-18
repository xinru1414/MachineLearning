"""
This program implements Naive Bayes for 
"""
import math
import pandas as pd
import statistics
import numpy as np
import operator
import sys


discrete_list = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "country"]
continuous_list = ["age", "fnlwgt", "education-num", "capital_gain" ,"capital_loss", "hoursperweek"]
q = 10 ** (-9)

class Model:
	def __init__(self, priors, discrete_param, continuous_param):
		self.priors = priors
		self.discrete_param = discrete_param
		self.continuous_param = continuous_param

		assert len(discrete_param) == len(discrete_list), "This is bad!!!"
		assert len(continuous_param) == len(continuous_list), "This is bad!!!"

def load_data(file_name):
	data_df = pd.read_csv(file_name, names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", 
		"relationship", "race", "sex", "capital_gain", "capital_loss", "hoursperweek", "country", "Class"])
	#print(len(data_df.index))
	return data_df

def clean_data(data_df):
	data_df = data_df[(data_df[["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", 
		"relationship", "race", "sex", "capital_gain", "capital_loss", "hoursperweek", "country", "Class"]] != " ?").all(axis=1)]
	#print(len(data_df.index))
	return data_df




def greater_data(data_df):
	return data_df[data_df["Class"] == " >50K"]


def less_data(data_df):
	return data_df[data_df["Class"] == " <=50K"]


def discrete_parameter(data_df):
	g_prob = []
	l_prob = []
	print("=======================================================================================")
	print("Discrete Feature\n")
	print("Class >50K")
	g_data_df = greater_data(data_df).filter(items=discrete_list)
	l_data_df = less_data(data_df).filter(items=discrete_list)

	for column in g_data_df:
		g = g_data_df[column].value_counts()/len(g_data_df.index)
		print(column + " " + str({k:round(v,4) for k, v in g.to_dict().items()}))
		g_prob.append({k:math.log(v) for k, v in g.to_dict().items()})


	print("\nClass <=50K")
	for column in l_data_df:
		l = l_data_df[column].value_counts()/len(l_data_df.index)
		print(column + " " + str({k:round(v, 4)for k, v in l.to_dict().items()}))
		l_prob.append({k:math.log(v) for k, v in l.to_dict().items()})
	
	return list(zip(g_prob, l_prob))

def discrete_prob(model, data_df):
	greater_prob, less_prob = zip(*model.discrete_param)

	discrete_df = data_df.filter(items=discrete_list)

	row_p = []

	for index, row in discrete_df.iterrows():
		l_prob = []
		g_prob = []
		for i in range(len(row)):
			if row[i] in less_prob[i].keys():
				l_prob.append(less_prob[i].get(row[i]))
			if row[i] not in less_prob[i].keys():
				l_prob.append(float("-inf"))
			if row[i] in greater_prob[i].keys():
				g_prob.append(greater_prob[i].get(row[i]))
			if row[i] not in greater_prob[i].keys():
				g_prob.append(float("-inf"))

		p_l_xy = sum(l_prob)
		p_g_xy = sum(g_prob)
		p_xy_d = (p_g_xy,p_l_xy)
		row_p.append(p_xy_d)
	discrete_df["p_xy_d"] = row_p

	return discrete_df


def continuous_parameter(data_df):
	greater_parameters = []
	less_parameters = []
	print("=======================================================================================")
	print("Continuous Feature")
	print("Class >50K")
	g_data = greater_data(data_df).filter(items=continuous_list)
	for column in g_data:
		mean = round(statistics.mean(list(map(int,g_data[column]))),4)
		variance = round(statistics.variance(list(map(int,g_data[column]))),4)
		greater_parameters.append((mean, variance))
		# report mean and variance
		print(column + " " + str(mean)+ " " + str(variance))

	print("Class <=50K")
	l_data = less_data(data_df).filter(items=continuous_list)
	for column in l_data:
		mean = round(statistics.mean(list(map(int,l_data[column]))),4)
		variance = round(statistics.variance(list(map(int,l_data[column]))),4)
		less_parameters.append((mean, variance))
		# report mean and variance
		print(column + " " + str(mean)+ " " + str(variance))

	return list(zip(greater_parameters, less_parameters))

def continuous_prob(model, data_df):
	greater_prob, less_prob = zip(*model.continuous_param)

	continuous_df = data_df.filter(items=continuous_list)
	row_p = []
	for index, row in continuous_df.iterrows():
		p = []
		for i in range(len(row)):
			# greater_prob
			#print(row[i], -(row[i] - less_prob[i][0])**2, (2 * (less_prob[i][1] + q)), -(row[i] - less_prob[i][0])**2 / (2 * (less_prob[i][1] + q)),1 / math.sqrt(2 * math.pi * (less_prob[i][1] + q)) * math.exp(-(row[i] - less_prob[i][0])**2 / (2 * (less_prob[i][1] + q))))
			p_l = math.log(1 / math.sqrt(2 * math.pi * (less_prob[i][1] + q))) + (-(row[i] - less_prob[i][0])**2 / (2 * (less_prob[i][1] + q)))
			#p_l = math.log(-(row[i] - less_prob[i][0])**2 / (2 * (less_prob[i][1] + q))) #math.log(1 / math.sqrt(2 * math.pi * (less_prob[i][1] + q))) 
			p_g = math.log(1 / math.sqrt(2 * math.pi * (greater_prob[i][1] + q))) + (-(row[i] - greater_prob[i][0])**2 / (2 * (greater_prob[i][1] + q)))
			# less prob
			#p_l = math.log(1 / math.sqrt(2 * math.pi * (less_prob[i][1] + q)) * math.exp(-(row[i] - less_prob[i][0])**2 / (2 * (less_prob[i][1] + q))))
			p.append((p_g, p_l))
		#print(p)
		p_g_xy = sum([i[0] for i in p])
		p_l_xy = sum([i[1] for i in p])
		p_xy = (p_g_xy, p_l_xy)
		row_p.append(p_xy)
	continuous_df["p_xy_c"] = row_p
	#print(continuous_df)
	return continuous_df

def log_posterior(model, data_df):
	v_1 = discrete_prob(model, data_df)["p_xy_d"].tolist()
	v_2 = continuous_prob(model, data_df)["p_xy_c"].tolist()

	v_3 = [(c+d, e+h) for (c, e), (d, h) in zip(v_1, v_2)]

	#print(v_3)

	data_df["p_xy"] = v_3
	return data_df


def prior(train_df):
	greater = greater_data(train_df)
	less = less_data(train_df)

	count_greater = len(greater)
	count_less = len(less)

	p_greater = float(count_greater) / float(len(train_df))
	p_less = float(count_less) / float(len(train_df))

	print("Prior >50k, <=50k " + str(p_greater) + " " + str(p_less))
	return (p_greater, p_less)

def predict(model, data_df):
	v_1 = log_posterior(model, data_df)["p_xy"].tolist()
	v_2 = model.priors
	v_3 = []

	for i in range(len(v_1)):
		v = tuple(map(operator.add, v_1[i], v_2))
		v_3.append(v)

	data_df["pos"] = v_3
	return data_df

def make_prediction(data_df):
	predicted = []
	for index, row in data_df.iterrows():
		# g_prob > l_prob
		if row["pos"][0] > row["pos"][1]:
			predicted.append(" >50K")
		else:
			predicted.append(" <=50K")
	data_df["Predicted Class"] = predicted
	return data_df

def accuracy(data_df):
	print(data_df)
	correct_count = 0
	for i in range(len(data_df)):
		if data_df['Class'][i] == data_df["Predicted Class"][i]:
			correct_count+=1
	accuracy = round(correct_count / len(data_df.index), 4)
	print(accuracy)
	return accuracy

def training(data_df):
	priors = prior(data_df)
	discrete_param = discrete_parameter(data_df)
	continuous_param = continuous_parameter(data_df)
	# log_posterior(data_df)
	# predict(data_df)
	# make_prediction(data_df)
	# accuracy(data_df)
	return Model(priors, discrete_param, continuous_param)

def testing(model, data_df):
	#print("Testing...")
	#print("\tpredicting...")
	predict(model, data_df)
	#print("\tmaking prediction...")
	make_prediction(data_df)
	#print("\taccuracying...")
	accuracy(data_df)

def main():
	train_file_name = sys.argv[1]
	test_file_name = 'adult.test'
	
	train_df = load_data(train_file_name)
	test_df = load_data(test_file_name)

	clean_train_df = clean_data(train_df)

	print("=======================================================================================")
	print("Training")
	model = training(clean_train_df)

	print("=======================================================================================")
	print("Testing with train_df")
	testing(model, train_df)
	
	print("=======================================================================================")
	print("Testing with test_df")
	testing(model, test_df)

if __name__ == '__main__':
	main() 
