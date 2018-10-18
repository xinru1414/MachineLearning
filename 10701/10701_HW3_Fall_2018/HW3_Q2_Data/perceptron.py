"""
This program implements a simple perceptron algorithm
Xinru Yan
Oct 2018

Usage:
    python perceptron.py Xtrain.txt ytrain.txt Xtest.txt ytest.txt
"""
import sys
import numpy as np


def read_file(file_name):
    lines = []
    with open(file_name, 'r') as fp:
        for line in fp.readlines():
            line = line.strip().split()
            line = [float(x) for x in line]
            lines.append(line)
    return lines

def train(x: list, y: list):
    x = x
    y = y

    # initialize the weight
    w = np.zeros((10,))
    #w = [0,0,0,0,0,0,0,0,0,0]
    # initialize the bias
    b = 1

    i = 0

    correct = False
    while (i < 1) and not correct:
        correct = True
        for (x_value, y_value) in zip(x, y):
            if ((np.dot(w, x_value) + b) * y_value[0]) <= 0:
                correct = False
                b = b + y_value[0]
                w = np.add(np.multiply(y_value[0], x_value), w)
        i += 1
    return w, b, i

def test(x_train: list, y_train:list, x_test:list, y_test:list):
    x_train = x_train
    y_train = y_train

    x_test = x_test
    y_test = y_test

    print("test case numbers " + str(len(y_test)))

    p = train(x_train, y_train)
    w = p[0]
    b = p[1]
    count = 0
    for x_value, y_value in zip (x_test, y_test):
        if ((np.dot(w, x_value) + b) * y_value[0]) >= 0:
            count += 1
    print("correct prediction numbers " + str(count))
    print("accuracy " + str(count/len(y_test)))
    return count/len(y_test)


def main():
    train_x = sys.argv[1]
    train_y = sys.argv[2]
    test_x = sys.argv[3]
    test_y = sys.argv[4]
    print("Training")
    #print("X")
    x = read_file(train_x)
    #print("Y")
    y = read_file(train_y)

    print(train(x,y))
    print("Training and Testing")
    x_t = read_file(test_x)
    y_t = read_file(test_y)
    test(x, y, x_t, y_t)



if __name__ == '__main__':
    main()



