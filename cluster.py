
m sklearn.cluster import MiniBatchKMeans
from scipy import array, sparse
from Parsing import save_matrix

def cluster(train_data, test_data, tag_matrix, k):
    km = MiniBatchKMeans(k)
    training_labels = km.fit_predict(train_data)
    testing_labels = km.predict(test_data)
    training_matrices = []
    testing_matrices = []
    tag_matrices = []
    for i in xrange(k):
        train_rows = [train_data.getrow(j) for j in xrange(train_data.shape[0]) if training_labels[j] == i]
        test_rows = [test_data.getrow(j) for j in xrange(test_data.shape[0]) if testing_labels[j] == i]
        if len(train_rows) == 0:
            training_matrices.append(sparse.csr_matrix((1,1)))
            testing_matrices.append(sparse.csr_matrix((1,1)))
            tag_matrices.append(tag_matrix.getrow(0)-tag_matrix.getrow(0))
            continue
        training_matrices.append(sparse.vstack(train_rows))
        if len(test_rows) == 0:
            testing_matrices.append(sparse.csr_matrix((1,1)))
            tag_matrices.append(tag_matrix.getrow(0)-tag_matrix.getrow(0))
            continue
        testing_matrices.append(sparse.vstack(test_rows))
        ktags = sum([tag_matrix.getrow(j) for j in xrange(tag_matrix.shape[0]) if training_labels[j] == i])
        for j in xrange(len(ktags.data)):
            ktags.data[j] /= ktags.data[j]
        tag_matrices.append(ktags)
    tag_matrix = sparse.vstack(tag_matrices)
    save_matrix('tag_matrix.txt', tag_matrix)
    for i in xrange(k):
        save_matrix('training_matrix_%d.txt' % i, training_matrices[i].tocoo())
        save_matrix('testing_matrix_%d.txt' % i, testing_matrices[i].tocoo())
    
    predictions = []
    for i in xrange(len(testing_labels)):
        predictions.append(tag_matrices[testing_labels[i]])
    return sparse.vstack(predictions)

