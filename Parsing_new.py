from numpy import *
from scipy.sparse import *
from HTMLParser import HTMLParser
import re
import string
import itertools
import sys
import time
import threading
from collections import Counter
import cPickle
import cProfile, pstats, StringIO
from multiprocessing import Process, Queue
from scipy.io import savemat





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
        print 'Saving ' + filename + '...'
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

                        if v != 0:
                            mat_writer.write("%d %d %s" % ((i-start_rows),j,v) + '\n')


def load_matrix(filename):
        with open(filename, 'rU') as mat_reader:
                line = mat_reader.readline().split()
                numRows = int(line[0])
                numCols = int(line[1])
                #matrix = lil_matrix((numRows, cols))
                vals = []
                rows = []
                cols = []
                for line in mat_reader:
                        line = line.split()
                        row = int(line[0])
                        col = int(line[1])
                        val = float(line[2])
                        vals.append(val)
                        rows.append(row)
                        cols.append(col)

                        #matrix[numRows, numCols] = val
                return coo_matrix((vals,(rows, cols)), shape=(numRows, numCols)).tocsr()

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

        def __init__(self, training_filename, testing_filename, max_train_lines=0, max_test_lines=0, stop_filename='', n=2, tf_idf=True, combine_code=True, check_substring_tags=False, use_words_map=False, num_threads=10, skip_preprocessing=False):
                self.ngram_map = {}
                self.ngram_list = []
                self.tags = set()
                self.tag_list = []
                self.tag_map = {}
                self.num_tags = 0
                self.training_set = []
                self.test_set = []
                self.ngram_set = []
                self.next_index = 0
                self.training_question_index = 0
                self.testing_question_index = 0
                self.training_questions = 0
                self.testing_questions = 0
                self.total_questions = 0
                self.tf_idf = tf_idf
                self.post_tf_idf = False
                self.tf_idf_map = Counter()
                self.use_words_map = use_words_map
                self.words_map = {}
                self.stop_words = set()
                self.n = n
                self.combine_code = combine_code
                self.max_train_lines = max_train_lines
                self.max_test_lines = max_test_lines
                self.check_substring_tags = check_substring_tags
                self.log_list = []
                self.num_ngrams = 0
                self.list_training_bag = [[]]
                self.list_testing_bag = [[]]
                self.list_training_tags = [[]]
                self.training_question_list = []
                self.testing_question_list = []
                self.num_threads = num_threads
                self.threads = []
                self.training_filename = training_filename
                self.testing_filename = testing_filename
                self.stop_filename = stop_filename
                self.skip_preprocessing = skip_preprocessing
                self.initialize()

        def initialize(self): 
                print 'Running with training data: ' + self.training_filename + ' (max ' + str(self.max_train_lines) + ' lines), testing data: ' + self.testing_filename + ' (max ' + str(self.max_test_lines) + ' lines)'
                print 'Options: stop file: ' + self.stop_filename + ', max ngram length: ' + str(self.n) + ', tf-idf: ' + str(self.tf_idf) + ', combining code: ' + str(self.combine_code) + ', checking substring tags: ' + str(self.check_substring_tags)
                if not self.skip_preprocessing or True:
                    self.build_stop_set(self.stop_filename)
                    self.build_tag_list(self.training_filename)

                self.build_bag_of_words(self.training_filename, self.testing_filename, self.n)

                #if not self.skip_preprocessing :
                #    return

                if self.post_tf_idf:
                        self.training_bag, self.testing_bag = tf_id(self.training_bag, self.testing_bag)

                self.training_bag = self.training_bag.tocoo()
                #self.testing_bag = self.testing_bag.tocoo()
                self.training_tags = self.training_tags.tocoo()

                processes = []
                # processes.append(Process(target=save_matrix, args=('testing_matrix_full.txt', self.testing_bag)))
                # processes[-1].start()
                # del self.testing_bag

                processes.append(Process(target=save_matrix, args=('training_matrix_full.txt', self.training_bag)))
                processes[-1].start()

                processes.append(Process(target=save_matrix, args=('training_matrix.txt', self.training_bag, 0, int(self.max_train_lines * .7))))
                processes[-1].start()
                # processes.append(Process(target=save_matrix, args=('testing_matrix.txt', self.training_bag, int(self.max_train_lines * .7), 0)))
                # processes[-1].start()
                #del self.training_bag
                
                processes.append(Process(target=save_matrix, args=('training_tag_matrix_full.txt', self.training_tags)))
                processes[-1].start()
                processes.append(Process(target=save_matrix, args=('training_tags.txt', self.training_tags, 0, int(self.max_train_lines * .7))))
                processes[-1].start()

                save_matrix('testing_tags.txt', self.training_tags, int(self.max_train_lines * .7), 0)
                #del self.training_tags
                self.write_log('log.txt')

                for proc in processes:
                    proc.join()

                processes = []

                processes.append(Process(target=savemat, args=('training_tag_matrix_full_matlab.mat',  {'tags':self.training_tags})))
                processes[-1].start()

                processes.append(Process(target=savemat, args=('training_matrix_full_matlab.mat', {'train':self.training_bag})))
                processes[-1].start()


                for proc in processes:
                    proc.join()
                print load_matrix("training_matrix.txt")
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

                if self.use_words_map:
                    self.words_map[qID] = words

                if len(words) > 5000:
                    raise NameError('Question too long')

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
                                if not ngram in tf_temp_set:
                                    if training:
                                        self.training_set.append(ngram)
                                    else:
                                        self.test_set.append(ngram)

                                if self.tf_idf and not ngram in tf_temp_set:
                                        self.tf_idf_map.update(Counter({ngram:1}))
                                        tf_temp_set.add(ngram)


        def process_question_again(self, question, n, training=True, thread_id=0, que=None):
                mat = 0
                question = question.replace('"""""', '"')
                question = question.replace('""', ' ')
                question = question.replace('", "', '","')
                question = question.split('","')

                qID = int(question[0][1:])

                words = []
                if self.use_words_map and qID in self.words_map:
                    words = self.words_map[qID]
                else:
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

                if len(words) > 5000:
                    raise NameError('Question too long')

                csr_data = []
                csr_rows = []
                csr_cols = []
                vals = {}

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

                                    if not ngram_index in vals:
                                        csr_rows.append(0)
                                        csr_cols.append(ngram_index)
                                        csr_data.append(inc_value)
                                        vals[ngram_index] = len(csr_data) - 1
                                    else:
                                        csr_data[vals[ngram_index]] += inc_value

                if training:
                        self.list_training_bag.append(csr_matrix((csr_data, (csr_rows, csr_cols)), shape=(1,self.num_ngrams)))
                else:
                        self.list_testing_bag.append(csr_matrix((csr_data, (csr_rows, csr_cols)), shape=(1,self.num_ngrams)))


                if training:
                        qTags = question[3].lower()[:-1]
                        qTags = qTags.split()
                        tag_rows = []
                        tag_cols = []
                        tag_data = []
                        seen_tags = set()
                        for qTag in qTags:
                                if qTag in seen_tags:
                                    continue

                                seen_tags.add(qTag)

                                if qTag in self.tag_map:
                                        tag_index = self.tag_map[qTag]
                                        tag_rows.append(0)
                                        tag_cols.append(tag_index)
                                        tag_data.append(1)
                                else:
                                        self.log('Tag: ' + qTag + ', in question: ' + str(qID) + ' (' + qTitle + ') not found in tag list!')
                        self.list_training_tags.append(csr_matrix((tag_data, (tag_rows, tag_cols)), shape=(1,self.num_tags)))


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


        def parse_training_thread(self, start_ind, end_ind, thread_id, que):
            del self.testing_question_list

            num_read = 0
            self.list_training_bag = []
            self.list_training_tags = []
            for i in xrange(start_ind, end_ind):
                question = self.training_question_list[i]
                num_read += 1
                if num_read % 1000 == 0:
                        print 'Thread ' + str(thread_id) + ' at: ' + str(num_read)
                try:
                    self.process_question_again(question, self.n, True, thread_id)
                    self.training_question_index += 1
                except:
                    print 'Error parsing question: ' + str(num_read) 
                    print question

                if num_read % 500 == (500 / self.num_threads) * thread_id:
                    que.put((True, thread_id, self.list_training_bag, self.list_training_tags))
                    del self.list_training_bag
                    del self.list_training_tags
                    self.list_training_tags = []
                    self.list_training_bag = []

            que.put((False, thread_id, self.list_training_bag, self.list_training_tags))

        def parse_testing_thread(self, start_ind, end_ind, thread_id, que):
            num_read = 0
            self.list_testing_bag = []
            for i in xrange(start_ind, end_ind):
                question = self.testing_question_list[i]
                num_read += 1
                if num_read % 1000 == 0:
                        print 'Thread ' + str(thread_id) + ' at: ' + str(num_read)
                try:
                    self.process_question_again(question, self.n, False, thread_id)
                    self.testing_question_index += 1
                except:
                    print 'Error parsing question: ' + str(num_read) 
                    print question

                if num_read % 500 == (500 / self.num_threads) * thread_id:
                    que.put((True, thread_id, self.list_testing_bag))
                    del self.list_testing_bag
                    self.list_testing_bag = []

            que.put((False, thread_id, self.list_testing_bag))


        def preprocess_training_thread(self, start_ind, end_ind, thread_id, que):
            del self.testing_question_list

            num_read = 0
            for i in xrange(start_ind, end_ind):
                question = self.training_question_list[i]
                num_read += 1
                if num_read % 1000 == 0:
                        print 'Thread ' + str(thread_id) + ' at: ' + str(num_read)
                try:
                    self.process_question(question, self.n, True)
                except:
                    print 'Error parsing question: ' + str(num_read) 
                    print question

                if num_read % 500 == (500 / self.num_threads) * thread_id:
                    que.put((True, thread_id, self.tf_idf_map, self.training_set))
                    del self.tf_idf_map
                    del self.training_set
                    self.tf_idf_map = Counter()
                    self.training_set = []

            que.put((False, thread_id, self.tf_idf_map, self.training_set))

        def preprocess_testing_thread(self, start_ind, end_ind, thread_id, que):
            del self.training_question_list

            num_read = 0
            for i in xrange(start_ind, end_ind):
                question = self.testing_question_list[i]
                num_read += 1
                if num_read % 1000 == 0:
                        print 'Thread ' + str(thread_id) + ' at: ' + str(num_read)
                try:
                    self.process_question(question, self.n, False)
                except:
                    print 'Error parsing question: ' + str(num_read) 
                    print question

                if num_read % 500 == (500 / self.num_threads) * thread_id:
                    que.put((True, thread_id, self.tf_idf_map, self.test_set))
                    del self.tf_idf_map
                    del self.test_set
                    self.tf_idf_map = Counter()
                    self.test_set = []

            que.put((False, thread_id, self.tf_idf_map, self.test_set))



        def build_bag_of_words(self, training_filename, testing_filename, n=2):
                if not self.skip_preprocessing or True:
                    print 'Reading training file...'
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
                                            try:
                                                self.training_question_list.append(question)
                                                self.training_questions += 1
                                            except:
                                                print 'Error processing question: ' + str(expected_num) 
                                                print question 
                                            question = line
                                            expected_num += 1
                                    else:
                                            question += ' ' + line

                    print 'Reading testing file...'
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
                                            try:
                                                self.testing_question_list.append(question[:-1])
                                                self.testing_questions += 1
                                            except:
                                                print 'Error processing question: ' + str(expected_num) 
                                                print question
                                            question = line
                                            expected_num += 1
                                    else:
                                            question += ' ' + line


                    print 'Preprocessing training file...'
                    num_questions = len(self.training_question_list)
                    questions_per_thread = num_questions / self.num_threads
                    next_start = 0
                    current_thread = 0
                    thread_queue = Queue()
                    for thread_num in xrange(self.num_threads):
                        next_end = next_start + questions_per_thread
                        if thread_num == self.num_threads - 1:
                            next_end = num_questions

                        self.threads.append(Process(target=BagOfWords.preprocess_training_thread, args=(self, next_start, next_end, current_thread, thread_queue)))
                        self.threads[thread_num].start()
                        next_start += questions_per_thread
                        current_thread += 1


                    thread_count = 0
                    count = 0
                    while thread_count != self.num_threads:
                        flag, thread_id, tf_idf_to_add, set_to_add = thread_queue.get()
                        if flag == False:
                            thread_count += 1
                        self.tf_idf_map.update(tf_idf_to_add)

                        self.training_set.extend(set_to_add)
                        del set_to_add
                        del tf_idf_to_add
                        if count % 100 == 0:
                            print 'Master thread at: ' + str(count)
                        
                        count += 1


                    for threadToJoin in self.threads:
                        threadToJoin.join()


                    self.threads = []

                    # print 'Preprocessing testing file...'
                    # num_questions = len(self.testing_question_list)
                    # questions_per_thread = num_questions / self.num_threads
                    # next_start = 0
                    # current_thread = 0
                    # thread_queue = Queue()
                    # for thread_num in xrange(self.num_threads):
                    #     next_end = next_start + questions_per_thread
                    #     if thread_num == self.num_threads - 1:
                    #         next_end = num_questions

                    #     self.threads.append(Process(target=BagOfWords.preprocess_testing_thread, args=(self, next_start, next_end, current_thread, thread_queue)))
                    #     self.threads[thread_num].start()
                    #     next_start += questions_per_thread
                    #     current_thread += 1

                    # count = 0
                    # thread_count = 0
                    # while thread_count != self.num_threads:
                    #     flag, thread_id, tf_idf_to_add, set_to_add = thread_queue.get()
                    #     if flag == False:
                    #         thread_count += 1
                    #     self.tf_idf_map.update(tf_idf_to_add)

                    #     self.test_set.extend(set_to_add)
                    #     del set_to_add
                    #     del tf_idf_to_add
                    #     if count % 100 == 0:
                    #         print 'Master thread at: ' + str(count)

                    #     count += 1

                    # for threadToJoin in self.threads:
                    #     threadToJoin.join()

                    self.threads = []

                    self.total_questions = self.training_questions + self.testing_questions
                    self.ngram_set = set(self.training_set).union(set(self.test_set))
                    
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

                    self.num_ngrams = next_index

                    # with open('pickledBag', 'w') as picklefile:
                    #     pickler = cPickle.Pickler(picklefile)
                    #     pickler.dump(self)
                    # return

                

                self.training_bag = 0
                self.testing_bag =  0
                self.training_tags = 0

                print 'Parsing training file...'
                num_questions = len(self.training_question_list)
                questions_per_thread = num_questions / self.num_threads
                next_start = 0
                current_thread = 0
                thread_queue = Queue()
                for thread_num in xrange(self.num_threads):
                    next_end = next_start + questions_per_thread
                    if thread_num == self.num_threads - 1:
                        next_end = num_questions

                    self.threads.append(Process(target=BagOfWords.parse_training_thread, args=(self, next_start, next_end, current_thread, thread_queue)))
                    self.threads[thread_num].start()
                    self.list_training_bag.append([])
                    self.list_training_tags.append([])
                    next_start += questions_per_thread
                    current_thread += 1

                thread_count = 0
                count = 0
                while thread_count != self.num_threads:
                    flag, thread_id, bag_to_add, tags_to_add = thread_queue.get()
                    if flag == False:
                        thread_count += 1
                    self.list_training_bag[thread_id].extend(bag_to_add)
                    self.list_training_tags[thread_id].extend(tags_to_add)
                    if count % 100 == 0:
                        print 'Master thread at: ' + str(count)

                    count += 1
                    
                for threadToJoin in self.threads:
                    threadToJoin.join()

                bag = list(itertools.chain.from_iterable(self.list_training_bag))
                self.training_bag = vstack(bag)
                del self.list_training_bag
                del bag

                tag_bag = list(itertools.chain.from_iterable(self.list_training_tags))
                self.training_tags = vstack(tag_bag)
                del self.list_training_tags
                del tag_bag
                del self.training_question_list

                self.threads = []

                # print 'Parsing testing file...'
                # num_questions = len(self.testing_question_list)
                # questions_per_thread = num_questions / self.num_threads
                # next_start = 0
                # current_thread = 0
                # thread_queue = Queue()
                # for thread_num in xrange(self.num_threads):
                #     next_end = next_start + questions_per_thread
                #     if thread_num == self.num_threads - 1:
                #         next_end = num_questions

                #     self.threads.append(Process(target=BagOfWords.parse_testing_thread, args=(self, next_start, next_end, current_thread, thread_queue)))
                #     self.threads[thread_num].start()
                #     self.list_testing_bag.append([])
                #     next_start += questions_per_thread
                #     current_thread += 1

                # thread_count = 0
                # count = 0
                # while thread_count != self.num_threads:
                #     flag, thread_id, bag_to_add = thread_queue.get()
                #     if flag == False:
                #         thread_count += 1
                #     self.list_testing_bag[thread_id].extend(bag_to_add)
                #     if count % 100 == 0:
                #         print 'Master thread at: ' + str(count)

                #     count += 1

                # for threadToJoin in self.threads:
                #     threadToJoin.join()
               

                # bag = list(itertools.chain.from_iterable(self.list_testing_bag))
                # self.testing_bag = vstack(bag)
                del self.list_testing_bag
                #del bag

                del self.ngram_map
                del self.tags
                del self.tag_map
                del self.tf_idf_map
                del self.stop_words

                self.training_bag = self.training_bag.tocsr()
                # self.testing_bag = self.testing_bag.tocsr()

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
        #cProfile.run("bag = BagOfWords('Train.txt', 'Test.txt', 5000, 5000, 'stop_words.txt', n=2, tf_idf=True, combine_code=True, check_substring_tags=False)")
        bag = BagOfWords('Train.txt', 'Test.txt', 10000, 1, 'stop_words.txt', n=2, tf_idf=True, combine_code=True, check_substring_tags=False, skip_preprocessing=False)
        # with open('pickledBag', 'r') as picklefile:
        #     bag = cPickle.load(picklefile)
        #     bag.skip_preprocessing = True
        #     bag.num_threads = 10
        #     bag.initialize()

if __name__ == "__main__":
    main()
