from numpy import *
from scipy.sparse import *
from HTMLParser import HTMLParser
import re
import string
import itertools
import sys
import time

#Thanks Eloff from StackOverflow!
class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    html = html.decode("ascii","ignore").encode("ascii")
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def save_matrix(filename, matrix, start_rows=0, end_rows=0):
        with open(filename, 'w') as mat_writer:
                rows = matrix.shape[0]
                if end_rows != 0:
                        rows = end_rows
                if start_rows != 0:
                        rows -= start_rows

                mat_writer.write(str(rows) + ' ' + str(matrix.shape[1]) + '\n')
                for i,j,v in itertools.izip(matrix.row, matrix.col, matrix.data):
                        if start_rows != 0 and i < start_rows:
                                continue

                        if end_rows != 0 and i >= end_rows:
                                break

                        mat_writer.write("%d %d %s" % ((i-start_rows),j,v) + '\n')


def load_matrix(filename):
        with open(filename, 'rU') as mat_reader:
                line = mat_reader.readline().split()
                rows = int(line[0])
                cols = int(line[1])
                matrix = lil_matrix((rows, cols))
                for line in mat_reader:
                        line = line.split()
                        row = int(line[0])
                        col = int(line[1])
                        val = float(line[2])
                        matrix[row, col] = val
                return matrix.tocsr()

def tf_id(training_data, testing_data):
        D = training_data.shape[0] + testing_data.shape[0]
        id_scores = zeros(training_data.shape[1])
        for i in xrange(training_data.shape[1]):
                count = 0
                for row in training_data:
                        if row[i] > 0:
                                count += 1
                for row in testing_data:
                        if row[i] > 0:
                                count += 1
                id_score = log(D / float(count))
                id_scores[i] = (id_score)

        for row in training_data:
                for i in xrange(len(row)):
                        row[i] *= id_scores[i]
        for row in testing_data:
                for i in xrange(len(row)):
                        row[i] *= id_scores[i]

        return (training_data, testing_data)

class BagOfWords(object):
        """docstring for BagOfWords"""

        def __init__(self, training_filename, testing_filename, max_train_lines=0, max_test_lines=0, stop_filename='', n=2, tf_idf=True, combine_code=True, check_substring_tags=False):
                self.ngram_map = {}
                self.ngram_list = []
                self.tags = set()
                self.tag_list = []
                self.tag_map = {}
                self.num_tags = 0
                self.training_set = set()
                self.test_set = set()
                self.ngram_set = set()
                self.next_index = 0
                self.training_question_index = 0
                self.testing_question_index = 0
                self.training_questions = 0
                self.testing_questions = 0
                self.total_questions = 0
                self.tf_idf = tf_idf
                self.post_tf_idf = False
                self.tf_idf_map = {}
                self.stop_words = set()
                self.n = n
                self.combine_code = combine_code
                self.max_train_lines = max_train_lines
                self.max_test_lines = max_test_lines
                self.check_substring_tags = check_substring_tags
                self.log_list = []

                print 'Running with training data: ' + training_filename + ' (max ' + str(max_train_lines) + ' lines), testing data: ' + testing_filename + ' (max ' + str(max_test_lines) + ' lines)'
                print 'Options: stop file: ' + stop_filename + ', max ngram length: ' + str(n) + ', tf-idf: ' + str(tf_idf) + ', combining code: ' + str(combine_code) + ', checking substring tags: ' + str(check_substring_tags)
                self.build_stop_set(stop_filename)
                self.build_tag_list(training_filename)
                self.build_bag_of_words(training_filename, testing_filename, self.n)

                if self.post_tf_idf:
                        self.training_bag, self.testing_bag = tf_id(self.training_bag, self.testing_bag)

                self.training_bag = self.training_bag.tocoo()
                self.testing_bag = self.testing_bag.tocoo()
                self.training_tags = self.training_tags.tocoo()

                save_matrix('testing_matrix_full.txt', self.testing_bag)
                del self.testing_bag

                save_matrix('training_matrix_full.txt', self.training_bag)
                save_matrix('training_matrix.txt', self.training_bag, 0, int(self.max_train_lines * .7))
                save_matrix('testing_matrix.txt', self.training_bag, int(self.max_train_lines * .7), 0)
                del self.training_bag
                
                save_matrix('training_tag_matrix_full.txt', self.training_tags)
                save_matrix('training_tags.txt', self.training_tags, 0, int(self.max_train_lines * .7))
                save_matrix('testing_tags.txt', self.training_tags, int(self.max_train_lines * .7), 0)
                del self.training_tags
                self.write_log('log.txt')
                #print load_matrix("training_matrix.txt")
                #print load_matrix("testing_matrix.txt")
                #print load_matrix("training_tag_matrix.txt")

        def log(self, message):
                self.log_list.append(message)

        def write_log(self, filename):
                with open(filename, 'w') as log_writer:
                        for message in self.log_list:
                                log_writer.write(message + '\n')

        def build_stop_set(self, stop_filename):
                if stop_filename == '':
                        return

                with open(stop_filename, 'rU') as stop_reader:
                        for line in stop_reader:
                                line = line.lower()
                                stops = line.split()
                                for word in stops:
                                        word = word.translate(string.maketrans("",""), string.punctuation)
                                        self.stop_words.add(word)


        def strip_code_and_formatting(self, input):
                codeList = re.split(r'<code>|</code>', input)
                text = ''
                code = ''
                if not self.combine_code:
                        text = ' '.join(codeList[0::2])
                        code = ' '.join(codeList[1::2])
                else:
                        text = ' '.join(codeList)

                text = strip_tags(text)
                text = text.translate(string.maketrans("",""), string.punctuation)
                return text.lower(), code


        def process_question(self, question, n, training=True):
                question = question.replace('"""""', '"')
                question = question.replace('""', ' ')
                question = question.replace('", "', '","')
                question = question.split('","')

                qID = int(question[0][1:])
                qTitle = question[1]
                qBody = question[2]

                qTitle, qTCode = self.strip_code_and_formatting(qTitle)
                qBody, qBCode = self.strip_code_and_formatting(qBody)
                body_words = qBody.split()
                title_words = qTitle.split()

                words = title_words + body_words

                new_words =[]
                for i, word in enumerate(words):
                        strip_punc = True

                        if self.check_substring_tags:
                                for tag in self.tags:
                                        if tag in word:
                                                strip_punc = False
                                                word = tag
                                                break
                        else:
                                if word in self.tags:
                                        strip_punc = False
                                elif word[:-1] in self.tags:
                                        word = word[:-1]
                                        strip_punc = False

                        if strip_punc:
                                word = word.translate(string.maketrans("",""), string.punctuation)
                        
                        if not word in self.stop_words:
                                new_words.append(word)

                words = new_words
                tf_temp_set = set()
                for index, word in enumerate(words):
                        for i in xrange(n+1):
                                if index + i >= len(words):
                                        break

                                ngram = ' '.join(words[index:(index + i)])
                                if training:
                                        self.training_set.add(ngram)
                                else:
                                        self.test_set.add(ngram)

                                if self.tf_idf and not ngram in tf_temp_set:
                                        if ngram in self.tf_idf_map:
                                                self.tf_idf_map[ngram] += 1
                                        else:
                                                self.tf_idf_map[ngram] = 1
                                        tf_temp_set.add(ngram)


        def process_question_again(self, question, n, training=True):
                question = question.replace('"""""', '"')
                question = question.replace('""', ' ')
                question = question.replace('", "', '","')
                question = question.split('","')

                qID = int(question[0][1:])
                qTitle = question[1]
                qBody = question[2]

                if training:
                        qTags = question[3].lower()[:-1]
                        qTags = qTags.split()
                        for qTag in qTags:
                                training_ind = self.training_question_index
                                if qTag in self.tag_map:
                                        tag_index = self.tag_map[qTag]
                                        self.training_tags[training_ind, tag_index] = 1
                                else:
                                        self.log('Tag: ' + qTag + ', in question: ' + str(qID) + ' (' + qTitle + ') not found in tag list!')

                qTitle, qTCode = self.strip_code_and_formatting(qTitle)
                qBody, qBCode = self.strip_code_and_formatting(qBody)

                body_words = qBody.split()
                title_words = qTitle.split()

                words = title_words + body_words

                new_words =[]
                for i, word in enumerate(words):
                        strip_punc = True

                        if self.check_substring_tags:
                                for tag in self.tags:
                                        if tag in word:
                                                strip_punc = False
                                                word = tag
                                                break
                        else:
                                if word in self.tags:
                                        strip_punc = False
                                elif word[:-1] in self.tags:
                                        word = word[:-1]
                                        strip_punc = False

                        if strip_punc:
                                word = word.translate(string.maketrans("",""), string.punctuation)
                        
                        if not word in self.stop_words:
                                new_words.append(word)

                words = new_words

                for index, word in enumerate(words):
                        for i in xrange(n+1):
                                if index + i >= len(words):
                                        break

                                ngram = ' '.join(words[index:(index + i)])
                                if ngram in self.ngram_map:
                                        ngram_index = self.ngram_map[ngram]

                                        inc_value = 1
                                        if self.tf_idf:
                                                inc_value = log(self.total_questions / float(self.tf_idf_map[ngram]))

                                        if training:
                                                question_ind = self.training_question_index
                                                self.training_bag[question_ind, ngram_index] += inc_value
                                        else:
                                                question_ind = self.testing_question_index
                                                self.testing_bag[question_ind, ngram_index] += inc_value


        def process_tags(self, question):
                question = question.replace('"""""', '"')
                question = question.replace('""', ' ')[1:]
                question = question.replace('", "', '","')
                question = question.split('","')

                qTags = []
                if len(question) > 3 and len(question[3]) >= 2:
                        qTags = question[3].lower()[:-2]
                        qTags = qTags.split()

                for tag in qTags:
                        self.tags.add(tag)


        def build_bag_of_words(self, training_filename, testing_filename, n=2):
                print 'Preprocessing training file...'
                with open(training_filename, 'rU') as data_reader:
                        data_reader.readline()
                        num_read = 0

                        question = data_reader.readline()
                        expected_num = question.split('"')[1]
                        print 'Starting id: ' + expected_num
                        expected_num = int(expected_num) + 1

                        for line in data_reader:
                                line = line.strip()
                                if line.find('"' + str(expected_num) + '"') != -1:
                                        if self.max_train_lines != 0 and num_read == self.max_train_lines:
                                                break

                                        num_read += 1
                                        if num_read % 1000 == 0:
                                                print num_read
                                        self.process_question(question, n, True)
                                        self.training_questions += 1
                                        question = line
                                        expected_num += 1
                                else:
                                        question += ' ' + line

                print 'Preprocessing testing file...'
                with open(testing_filename, 'rU') as data_reader:
                        data_reader.readline()
                        num_read = 0

                        question = data_reader.readline()
                        expected_num = question.split('"')[1]
                        print 'Starting id: ' + expected_num
                        expected_num = int(expected_num) + 1

                        for line in data_reader:
                                line = line.strip()
                                if line.find('"' + str(expected_num) + '"') != -1:
                                        if self.max_test_lines != 0 and num_read == self.max_test_lines:
                                                break

                                        num_read += 1
                                        if num_read % 1000 == 0:
                                                print num_read

                                        self.process_question(question[:-1], n, False)
                                        self.testing_questions += 1
                                        question = line
                                        expected_num += 1
                                else:
                                        question += ' ' + line

                self.total_questions = self.training_questions + self.testing_questions
                self.ngram_set = self.training_set.intersection(self.test_set)
                
                del self.training_set
                del self.test_set

                next_index = 0
                for ngram in self.ngram_set:
                        self.ngram_map[ngram] = next_index
                        self.ngram_list.append(ngram)
                        next_index += 1

                with open('ngram_list.txt', 'w') as ngram_writer:
                        ngram_writer.write('\n'.join(self.ngram_list))

                del self.ngram_list
                del self.ngram_set

                self.training_bag = lil_matrix((self.training_questions, next_index))
                self.testing_bag =  lil_matrix((self.testing_questions, next_index))
                self.training_tags = lil_matrix((self.training_questions, self.num_tags))

                print 'Parsing training file...'
                with open(training_filename, 'rU') as data_reader:
                        data_reader.readline()
                        num_read = 0

                        question = data_reader.readline()
                        expected_num = question.split('"')[1]
                        print 'Starting id: ' + expected_num
                        expected_num = int(expected_num) + 1

                        for line in data_reader:
                                line = line.strip()
                                if line.find('"' + str(expected_num) + '"') != -1:
                                        if self.max_train_lines != 0 and num_read == self.max_train_lines:
                                                break

                                        num_read += 1
                                        if num_read % 1000 == 0:
                                                print num_read

                                        self.process_question_again(question, n, True)
                                        self.training_question_index += 1
                                        question = line
                                        expected_num += 1
                                else:
                                        question += ' ' + line

                print 'Parsing testing file...'
                with open(testing_filename, 'rU') as data_reader:
                        data_reader.readline()
                        num_read = 0
                        
                        question = data_reader.readline()
                        expected_num = question.split('"')[1]
                        print 'Starting id: ' + expected_num
                        expected_num = int(expected_num) + 1

                        for line in data_reader:
                                line = line.strip()
                                if line.find('"' + str(expected_num) + '"') != -1:
                                        if self.max_test_lines != 0 and num_read == self.max_test_lines:
                                                break

                                        num_read += 1
                                        if num_read % 1000 == 0:
                                                print num_read

                                        self.process_question_again(question[:-1], n, False)
                                        self.testing_question_index += 1
                                        question = line
                                        expected_num += 1
                                else:
                                        question += ' ' + line

                del self.ngram_map
                del self.tags
                del self.tag_map
                del self.tf_idf_map
                del self.stop_words

                self.training_bag = self.training_bag.tocsr()
                self.testing_bag = self.testing_bag.tocsr()

        def build_tag_list(self, filename):
                print 'Building tag list...'
                with open(filename, 'rU') as tag_reader:
                        tag_reader.readline()
                        num_read = 0
                        question = ''
                        for line in tag_reader:
                                question += ' ' + line
                                if line.find('","') == 0:
                                        if self.max_train_lines != 0 and num_read == self.max_train_lines:
                                                break

                                        num_read += 1
                                        if num_read % 1000 == 0:
                                                print num_read

                                        self.process_tags(question)
                                        question = ''

                self.tag_list = []

                next_index = 0
                for tag in self.tags:
                        self.tag_map[tag] = next_index
                        self.tag_list.append(tag)
                        next_index += 1

                with open('tag_list.txt', 'w') as tag_writer:
                        tag_writer.write('\n'.join(self.tag_list))

                del self.tag_list

                self.num_tags = next_index

def main():
        bag = BagOfWords('Train.csv', 'Test.csv', 0, 0, 'stop_words.txt', 2)


if __name__ == "__main__":
    main()