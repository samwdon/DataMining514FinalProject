from sys import *
path.reverse()
from numpy import *
from scipy.sparse import *
from HTMLParser import HTMLParser
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import normalize
import re
import string
import itertools
import time
from Parsing import save_matrix, load_matrix

def knn(train, test, train_tags, test_tags, k=5, minCount=1):
	set_printoptions(threshold='nan')
	neigh = NearestNeighbors(k)
	print 'Fitting model'
	neigh.fit(train)
	print 'Running model on test data'
	distances, ind = neigh.kneighbors(test, return_distance=True)

	del distances

	print 'Tags shape: '
	print train_tags.shape

	output = lil_matrix((test.shape[0], train_tags.shape[1]))
	outputRow = 0

	for row in ind:
		rowSum = csr_matrix((1, train_tags.shape[1]))
		rowSumOutput = lil_matrix((1, train_tags.shape[1]))

		for index in row:
			rowSum = rowSum + train_tags.getrow(index)

		rowSumOutput = rowSum.tolil()
		for i in rowSum.nonzero()[1]:
			if rowSumOutput[0, i] >= minCount:
				output[outputRow, i] = 1

		outputRow += 1

	return output.tocsr()

def calcAccuracy(output_labels, test_labels):
	diff = output_labels - test_labels
	diff = diff.multiply(diff)
	size = diff.shape[0] * diff.shape[1]
	wrong = diff.sum() 

	accuracy = (1.0 - (float(wrong) / float(size)))
	falsePositive = (diff.multiply(output_labels)).sum()
	truePositive = output_labels.sum() - falsePositive
	falseNegative = diff.sum() - falsePositive
	trueNegative = size - falsePositive - falseNegative - truePositive
	return (accuracy, truePositive, falsePositive, trueNegative, falseNegative, wrong)

def printStats(output, test_tags):
	accuracy, truePositive, falsePositive, trueNegative, falseNegative, totalWrong = calcAccuracy(output, test_tags)
	print 'Raw accuracy (ignore this): ' + str(accuracy * 100)
	print 'true positive: ' + str(truePositive) 
	print 'false positive: ' + str(falsePositive)
	print 'true negative: ' + str(trueNegative)
	print 'false negative: ' + str(falseNegative)
	print 'sensitivity: ' + str(truePositive / float(truePositive + falseNegative))
	print 'specificity: ' + str(trueNegative / float(trueNegative + falsePositive))
	print 'Total wrong: ' + str(totalWrong)

def main():
	print 'Running knn...'
	train = load_matrix('training_matrix.txt')
	normalize(train, copy=False)
	print 'Loaded training data'
	test = load_matrix('testing_matrix.txt')
	normalize(test, copy=False)
	print 'Loaded testing data'

	train_tags = load_matrix('training_tags.txt')
	print 'Loaded training tags'
	test_tags = load_matrix('testing_tags.txt')
	print 'Loaded testing tags'

	output = knn(train, train, train_tags, train_tags, k=1, minCount=1)
	printStats(output, test_tags)

if __name__ == "__main__":
    main()