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
from Parsing_new import save_matrix, load_matrix, load_matrix_partial
from sklearn.random_projection import SparseRandomProjection
import multiprocessing as mp
from sklearn.linear_model import SGDClassifier
import traceback

def remove_singulars(mat, mat2, min_thresh=1):
	print 'Converting to csc'
	mat = mat.tocsc()
	mat2 = mat2.tocsc()
	output = []
	output2 = []
	currentCol = 0
	print 'Starting checks'
	for i in xrange(mat.shape[1]):
		if i % 100 == 0:
			print i
		newCol = mat.getcol(i)
		newCol2 = mat2.getcol(i)
		count = newCol.getnnz()
		if count > min_thresh:
			output.append(newCol)
			output2.append(newCol2)

	print 'Stacking...'
	return hstack(output).tocsr(), hstack(output2).tocsr()


def knn(train, test, train_tags, test_tags, k=5, minCount=1, Sparse=False):
	set_printoptions(threshold='nan')
	neigh = NearestNeighbors(k)
	distances = []
	ind= []

	print 'Fitting model, using'
	if Sparse:
		neigh.fit(train, 'ball_tree')
	else: 
		neigh.fit(train.todense(), 'ball_tree')

	print 'Running model on test data'
	if Sparse:
		distances, ind = neigh.kneighbors(test, return_distance=True)
	else:
		distances, ind = neigh.kneighbors(test.todense(), return_distance=True)

	del distances

	print 'Tags shape: '
	print train_tags.shape

	output = lil_matrix((test.shape[0], train_tags.shape[1]))
	rows = []
	cols = []
	vals = []
	outputRow = 0

	for row in ind:
		rowSum = csr_matrix((1, train_tags.shape[1]))
		rowSumOutput = lil_matrix((1, train_tags.shape[1]))

		for index in row:
			rowSum = rowSum + train_tags.getrow(index)

		rowSumOutput = rowSum.tolil()
		for i in rowSum.nonzero()[1]:
			if rowSumOutput[0, i] >= minCount:
				rows.append(outputRow)
				cols.append(i)
				vals.append(1)

		outputRow += 1

	return coo_matrix((vals, (rows, cols)), shape=(test.shape[0], train_tags.shape[1])).tocsr()

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

def sgd_helper(train, test, train_tags, test_tags, clf, start, end, thread_queue):
	rows = []
	cols = []
	vals = []

	count = 0

	for i in xrange(start, end):
		labels = ravel(train_tags.getcol(i).todense())
		if labels.sum() < 10:
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
				rows.append(outputRow)
				cols.append(i)
				vals.append(1)
				#output[outputRow, i] = 1
			outputRow += 1

		if count % 100 == 0:
			thread_queue.put((True, rows, cols, vals))
			rows = []
			cols = []
			vals = []

		count += 1

	thread_queue.put((False, rows, cols, vals))



def sgd(train, test, train_tags, test_tags, c=10, ker='linear', gam=0.0, num_threads=3):
	train_tags = train_tags.tocsc()
	clf = SGDClassifier()

	if ker == 'log':
		clf = SGDClassifier(loss='log')

	output = lil_matrix((test.shape[0], train_tags.shape[1]))

	threads = []
	start = 0
	count = 0
	thread_queue = mp.Queue()
	current_thread = 0
	perThread = train_tags.shape[1] / num_threads

	rows = []
	cols = []
	vals = []

	for i in xrange(num_threads):
		end = start + perThread
		if i == num_threads - 1:
			next_end = train_tags.shape[1]
		threads.append(mp.Process(target=sgd_helper, args=(train, test, train_tags, test_tags, clf, start, end, thread_queue)))
		start += perThread

	for threadToStart in threads:
		threadToStart.start()

	count = 0
	num = 0
	while count != num_threads:
		flag, rowsAdd, colsAdd, valsAdd = thread_queue.get()
		if not flag:
			count += 1
		rows.extend(rowsAdd)
		cols.extend(colsAdd)
		vals.extend(valsAdd)
		if num % 100 == 0:
			print 'Master Thread at: ' + str(num)

	for threadToJoing in threads:
		threadToJoing.join()


	train_tags = train_tags.tocsr()
	return coo_matrix((vals, (rows, cols)), shape=(test.shape[0], train_tags.shape[1])).tocsr()




def applyPCA(train, test, target_dims=2500):
	pca1 = TruncatedSVD(n_components=target_dims)
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
	precision = (truePositive / (truePositive + falsePositive)) 
	recall = (truePositive / (truePositive + falseNegative))

	print 'Final score: ' + str(2 * (precision * recall) / (precision + recall) )

def main():


	print 'Loading full...'
	#full = load_matrix('training_matrix_full.txt')



	small, counts, newCols = load_matrix_partial('training_matrix_full.txt', 700000, min_thresh=4)

	print small.shape

	#tsvd = TruncatedSVD(5000)
	#tsvd.fit(small)


	print 'Running knn...'
	train = small #load_matrix('training_matrix.txt')
	normalize(train, copy=False)
	print 'Loaded training data'
	test, counts, newCols = load_matrix_partial('training_matrix_full.txt', 900000, 700000, min_thresh=4, use_dicts=True, counts=counts, newCols=newCols)
	normalize(test, copy=False)
	print 'Loaded testing data'

	#train, test = remove_singulars(train, test, 1)
	print train.shape

	#train = srp.transform(train)
	#test = srp.transform(test)

	#srp = SparseRandomProjection()
	#srp.fit(train)

	#train = srp.transform(train)
	#test = srp.transform(test)

	train_tags, counts, newCols = load_matrix_partial('training_tag_matrix_full.txt', 700000, min_thresh=10000)
	print 'Loaded training tags'
	test_tags, counts, newCols = load_matrix_partial('training_tag_matrix_full.txt', 900000, 700000, min_thresh=10000, use_dicts=True, counts=counts, newCols=newCols)
	print 'Loaded testing tags'

	print train_tags.shape


	svmOutput = []
	logOutput = []
	denseKNNOutput = []
	sparseKNNOutput = []
	sparseKNNOutput_transform = []
	svmOutput_transform = []
	logOutput_transform = []
	print 'Testing SVM...'
	try:
		svmOutput = sgd(train, test, train_tags, test_tags)
		printStats(svmOutput, test_tags)
	except Exception, err:
		print traceback.format_exc()

	print 'Testing logistic regression...'
	try:
		logOutput = sgd(train, test, train_tags, test_tags, ker='log')
		printStats(logOutput, test_tags)
	except Exception, err:
		print traceback.format_exc()

	# print 'Testing sparse KNN...'
	# try:
	# 	sparseKNNOutput = knn(train, test, train_tags, test_tags, k=7, minCount=3, Sparse=True)
	# 	printStats(sparseKNNOutput, test_tags)
	# except Exception, err:
	# 	print traceback.format_exc()


	# print 'Testing dense KNN with SVD and SRP...'
	# try:
		
	# 	train = load_matrix('training_matrix_full.txt', 100000)
	# 	train_tags = load_matrix('training_tag_matrix_full.txt', 100000)
	# 	srp = SparseRandomProjection()
	# 	srp.fit(train)
	# 	train = srp.transform(train)
	# 	test = srp.transform(test)
	# 	#qtrain, test = applyPCA(train, test)
	# 	denseKNNOutput = knn(train, test, train_tags, test_tags, k=7, minCount=3, Sparse=False)
	# 	printStats(denseKNNOutput, test_tags)
	# except Exception, err:
	# 	print traceback.format_exc()


	# print 'Testing sparse KNN with SRP...'
	# try:
	# 	sparseKNNOutput_transform = knn(train, test, train_tags, test_tags, k=7, minCount=3, Sparse=True)
	# 	printStats(sparseKNNOutput_transform, test_tags)
	# except Exception, err:
	# 	print traceback.format_exc()

	# print 'Testing SVM with SRP...'
	# try:
	# 	svmOutput_transform = sgd(train, test, train_tags, test_tags)
	# 	printStats(svmOutput_transform, test_tags)
	# except Exception, err:
	# 	print traceback.format_exc()

	# print 'Testing logistic regression with SRP...'
	# try:
	# 	logOutput_transform = sgd(train, test, train_tags, test_tags, ker='log')
	# 	printStats(logOutput_transform, test_tags)
	# except Exception, err:
	# 	print traceback.format_exc()


	# print 'Testing SVM...'
	# try:
	# 	printStats(svmOutput, test_tags)
	# except Exception, err:
	# 	print traceback.format_exc()

	# print 'Testing logistic regression...'
	# try:
	# 	printStats(logOutput, test_tags)
	# except Exception, err:
	# 	print traceback.format_exc()

	# print 'Testing sparse KNN...'
	# try:
	# 	printStats(sparseKNNOutput, test_tags)
	# except Exception, err:
	# 	print traceback.format_exc()


	# print 'Testing dense KNN with SVD and SRP...'
	# try:
	# 	printStats(denseKNNOutput, test_tags)
	# except Exception, err:
	# 	print traceback.format_exc()


	# print 'Testing sparse KNN with SVD and SRP...'
	# try:
	# 	printStats(sparseKNNOutput_transform, test_tags)
	# except Exception, err:
	# 	print traceback.format_exc()

	# print 'Testing SVM with SVD and SRP...'
	# try:
	# 	printStats(svmOutput_transform, test_tags)
	# except Exception, err:
	# 	print traceback.format_exc()

	# print 'Testing logistic regression with SVD and SRP...'
	# try:
	# 	printStats(logOutput_transform, test_tags)
	# except Exception, err:
	# 	print traceback.format_exc()



# printStats(output, test_tags)


# print 'Testing KNN...'
# output = knn(train, test, train_tags, test_tags, k=7, minCount=3)
# printStats(output, test_tags)


	#train, test = applyPCA(train, test)

	# print 'Testing KNN (w/ PCA)...'
	# output = knn(train, test, train_tags, test_tags, k=7, minCount=3)
	# printStats(output, test_tags)

	# print 'Testing SVM (w/ PCA)...'
	# output = svm(train, test, train_tags, test_tags)
	# printStats(output, test_tags)

if __name__ == "__main__":
    main()