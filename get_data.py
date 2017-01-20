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

# randomizes 3 related lists: the essays, their scores, and the essays POS tags
def randomize(essays, scores, POSfile):
    print 'start randomize'
    print "Len essays", len(essays)
    print "Len scores", len(scores)
    print "Len pos", len(POSfile)

    essays_unt = []
    scores_unt = []
    essays_tagged = []
    # print ess
    combined = list(zip(essays, scores, POSfile))
    i = 0 
  
    random.shuffle(combined)

    essays_unt[:], scores_unt[:], essays_tagged[:] = zip(*combined)
    return (essays_unt, scores_unt, essays_tagged)

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
        
        if essay_set != '2':
            grade = letter_grade(essay_set, essay_score)
            for word in essay:
                word_counts[word] += 1

            train_data_lines.append(essay)
            train_result_lines.append(grade)
            
            gradeCounts[grade] += 1

    train_data_lines = detect_unknowns(train_data_lines, word_counts)

    return (train_data_lines, train_result_lines, gradeCounts)

def print_to_file(data, file_name):
    f = open(file_name, "w+")

    for line in data:
        f.write(str(line))
        f.write('\n')

    f.close()

def main():

    f = open("training_set_rel3.tsv")
    lines = list(f)
    
    # pos_file = open("training_set_tagged.txt")
    pos_file = open("tagged_data_final.txt")
    pos_lines = list(pos_file)

    essays, scores, train_score_dict = get_training_data(lines)
    essays_randomized, scores_randomized, essays_tagged_randomized = randomize(essays, scores, pos_lines)
        
    print_to_file(essays_randomized, "essays_randomized.txt")
    print_to_file(scores_randomized, "scores_randomized.txt")
    
    f_tagged = open("essays_tagged_randomized.txt", 'w+')
    for line in essays_tagged_randomized:
        f_tagged.write(str(line))

    # print_to_file(essays_tagged_randomized, "essays_tagged_randomized.txt")


main()  