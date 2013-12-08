from sys import *
path.reverse()
from numpy import *
from scipy.sparse import *
from HTMLParser import HTMLParser
from sklearn.neighbors import NearestNeighbors, DistanceMetric
from sklearn.preprocessing import normalize
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import SparsePCA, PCA, TruncatedSVD
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

def svm(train, test, train_tags, test_tags, c=10, ker='linear', gam=0.0):
	train_tags = train_tags.tocsc()
	clf = SVC(C=c, kernel=ker, gamma=gam)

	if ker == 'linear':
		clf = LinearSVC(C=c)

	output = lil_matrix((test.shape[0], train_tags.shape[1]))

	for i in xrange(train_tags.shape[1]):
		labels = ravel(train_tags.getcol(i).todense())
		if labels.sum() == 0:
			print 'No instances of tag: ' + str(i)
			continue

		if i % 100 == 0:
			print str(i) + '/' + str(train_tags.shape[1])

		labels = (labels * 2) - 1

		clf.fit(train, labels)
		pred = clf.predict(test)
		outputRow = 0
		for val in pred:
			if val > 0:
				output[outputRow, i] = 1
			outputRow += 1

	train_tags = train_tags.tocsr()
	return output.tocsr()

def applyPCA(train, test, target_dims=2500):
	pca1 = TruncatedSVD(n_components=target_dims, algorithm='arpack')
	pca1.fit(train)
	train = pca1.transform(train)
	test = pca1.transform(test)
	return train, test

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

	print 'Testing KNN...'
	output = knn(train, test, train_tags, test_tags, k=7, minCount=3)
	printStats(output, test_tags)

	print 'Testing SVM...'
	output = svm(train, test, train_tags, test_tags)
	printStats(output, test_tags)

	#train, test = applyPCA(train, test)

	print 'Testing KNN (w/ PCA)...'
	output = knn(train, test, train_tags, test_tags, k=7, minCount=3)
	printStats(output, test_tags)

	print 'Testing SVM (w/ PCA)...'
	output = svm(train, test, train_tags, test_tags)
	printStats(output, test_tags)

if __name__ == "__main__":
    main()