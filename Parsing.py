from numpy import *
from HTMLParser import HTMLParser
import re

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
    s = MLStripper()
    s.feed(html)
    return s.get_data()

class Question:
	def __init__(self, qID, qTitle, qTiltle):




class BagOfWords(object):
	"""docstring for BagOfWords"""

	def strip_code_and_formatting(self, input):
		codeList = re.split(r'<code>|</code>', input)
		text = ' '.join(codeList[0::2])
		code = ' '.join(codeList[1::2])
		text = strip_tags(text)
		text = text.translate(string.maketrans("",""), string.punctuation)
		return text.lower(), code

	def process_question(self, question):
		question = question.replace('""', ' ')[1:]
		question = question.split('", "')
		qID = int(question[0])
		qTitle = question[1]
		qBody = question[2]




	def build_bag_of_words(self, filename):
		with open(filename, 'rU') as data_reader:
			question = ''
			for line in data_reader:
				if line == '"':
					process_question(question)
					question = ''
				else:
					question += ' ' + line




def main():
	tester 


main()