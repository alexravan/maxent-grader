# Naomi Zarrilli & Alex Ravan

import re
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np
import string
import nltk
from nltk.tag.stanford import StanfordPOSTagger
from nltk.internals import find_jars_within_path
from nltk import word_tokenize
import random
import ast

jar = '/Users/AlexanderRavan/GitHub/NLP-final/stanford-postagger-2015-12-09/stanford-postagger-3.6.0.jar'
model = '/Users/AlexanderRavan/GitHub/NLP-final/stanford-postagger-2015-12-09/models/english-left3words-distsim.tagger'

tagger = StanfordPOSTagger(model, jar)

rubric = { '1': 12, '3': 3, '4': 3, '5': 4, '6':4, '7': 30, '8':60 }
entities = {"organization": '<ORG>', 'location' : '<LOC>', 'caps':'<CAPS>', 
            'num': '<NUM>', 'percent': '<PERCENT>', 'person': '<PERS>', 
            'month': '<MONTH>', 'date': '<DATE>', 'money': '<MONEY>' }

UNKNOWN_TOKEN = "<UNK>"

# changes all words that occur less than twice to UNKNOWN token
def detect_unknowns(data, word_counts):
    data_with_unknowns = []

    for essay in data:
        new_essay = []
        for word in essay:
            word = word.lower()
            if word_counts[word] <= 1:
                new_essay.append(UNKNOWN_TOKEN)
            else:
                new_essay.append(word)
        data_with_unknowns.append(new_essay)
    
    return data_with_unknowns

# named entity recognition by changing named entities to tags
def NER(essay):
    to_remove = []
    for i in range(len(essay) - 1):
        try:
            if essay[i] == "@":
                entity = re.sub("\d+", "", essay[i+1])
                # print essay[i], essay[i+1], entity
                essay[i] = entities[entity]
                to_remove.append(essay[i+1])
        except KeyError:
            pass
    for i in to_remove:
        essay.remove(i)
    return essay
            
# assigns a letter grade to an essay based on the essay set and its numerical score
def letter_grade(essay_set, essay_score):
    essay_score_int = int(essay_score)

    if essay_set == '1':
        if essay_score == '5' or essay_score == '6':
            return 'A'
        elif essay_score == '4':
            return 'B'
        elif essay_score == '3':
            return 'C'
        elif essay_score == '2':
            return 'D'
        else:
            return 'F'
    elif essay_set == '3' or essay_set == '4':
        if essay_score == '3':
            return 'A'
        elif essay_score == '2':
            return 'B'
        elif essay_score == '1':
            return 'D'
        else:
            return 'F'
    elif essay_set == '5' or essay_set == '6':
        if essay_score == '4':
            return 'A'
        elif essay_score == '3':
            return 'B'
        elif essay_score == '2':
            return 'C'
        elif essay_score == '1':
            return 'D'
        else:
            return 'F'
    elif essay_set == '7':
        if essay_score_int >= 24:
            return 'A'
        elif essay_score_int >= 18:
            return 'B'
        elif essay_score_int >= 12:
            return 'D'
        else:
            return 'F'
    elif essay_set == '8':
        if essay_score_int >= 50:
            return 'A'
        elif essay_score_int >= 40:
            return 'B'
        elif essay_score_int >= 30:
            return 'C'
        elif essay_score_int >= 20:
            return 'D'
        else:
            return 'F'

# preprocessing training data by tokenizing each essay, splitting up
# essay and essay grade, adding in unknowns and named entity tags.
def get_training_data(lines):
    train_data_lines = []
    train_result_lines = []
    gradeCounts = defaultdict(int)
    word_counts = defaultdict(int)
    lines = lines[1:]

    for line in lines:

        line_data = re.split(r'\t+', line)
        line_data[2] = line_data[2].decode('unicode_escape').encode('ascii','ignore')
        
        essay = line_data[2].encode('utf-8')
        essay = word_tokenize(essay)
        essay = essay[1:-1] # remove quotes
        essay = NER(essay)  # replace NE with tokens
        essay_score = line_data[5]
        essay_set = line_data[1].encode('utf-8')
        
        # ignoring all essays from set 2
        if essay_set != '2':
            grade = letter_grade(essay_set, essay_score)
            for word in essay:
                word = word.lower()
                word_counts[word] += 1

            train_data_lines.append(essay)
            train_result_lines.append(grade)
            
            gradeCounts[grade] += 1

    train_data_lines = detect_unknowns(train_data_lines, word_counts)

    return (train_data_lines, train_result_lines, gradeCounts)

# returns set of all unique words in data
def get_vocabulary(data):
    vocabulary = set()

    for essay in data:
        for word in essay:
            word = word.lower()
            vocabulary.add(word)
    return vocabulary

def main():
	f = open("training_set_rel3.tsv")
	lines = list(f)
	essays, scores, train_score_dict = get_training_data(lines)
	
	train_data_essays, train_data_scores = essays, scores

	# POS_bash = "python ~/GitHub/NLP-final/nltk_cli/stanford.py --tool=postagger --jar=$HOME/stanford-parser/stanford-parser.jar --modeljar=$HOME/stanford-parser/stanford-parser-3.5.2-models.jar --model=edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz --input=essay.txt"
	# lex_bash = "python ~/GitHub/NLP-final/nltk_cli/stanford.py --tool=lexparser --jar=$HOME/stanford-postagger/stanford-postagger.jar --model=$HOME/stanford-postagger/models/english-bidirectional-distsim.tagger --input=essay.txt"
	# ner_bash = "python ~/GitHub/NLP-final/nltk_cli/stanford.py --tool=nertagger --jar=$HOME/stanford-ner/stanford-ner.jar --model=$HOME/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz --input=essay.txt"

	tagged_essays = open('all_tagged_data_dec21_7pm.txt', "w")
	for essay in train_data_essays:
		tagged_essays.write(str(tagger.tag(essay)))
		tagged_essays.write('\n')
	tagged_essays.close()

main()



