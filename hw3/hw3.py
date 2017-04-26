import numpy as np
import math, random, sys
import csv
from numpy import genfromtxt
from math import exp


def main():
	print 'Getting Data';
	Xtrain, Ytrain, Xtest, Ytest = getData()
	length = len(Xtest);

	print 'starting part 1.1';
	part1(Xtrain,Ytrain,Xtest,Ytest,1);
#	k = 1;
#	for k in range(51):
#		training_error, cross_validation, testing_error = part1(Xtrain,Ytrain,Xtest,Ytest,k)


def getData():
	training = genfromtxt('knn_train.csv', delimiter=',')
	testing = genfromtxt('knn_test.csv', delimiter=',')


	XTrain = np.array(training[:,1:30])
	YTrain = np.array(training[:,0:1])
	XTest = np.array(testing[:,1:30])
	YTest = np.array(testing[:,0:1])

	return XTrain, YTrain, XTest, YTest


if __name__ == '__main__':
	main()
