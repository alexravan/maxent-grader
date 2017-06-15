# Essay Grade Prediction Using a MaxEnt Classifier

##### Naomi Zarrilli and Alexander Ravan 
##### COMP150-04: Natural Language Processing 

## Abstract

We took on the task of predicting essay grades. We attempted to solve this problem using MaxEnt. Our data is from Kaggle (https://www.kaggle.com/c/asap-aes/data). It is labelled with numerical scores that vary depending on the essay set. There are eight sets and 11178 essay total. For our most accurate model, we used part of speech tags and tfidf as features. Our baseline was bag of words with Naive Bayes classification. Our baseline accuracy was 45.9%. We were able to increase our accuracy by nearly 20% using part of speech tags, bigrams, and tfidf as features for MaxEnt. Our resulting accuracy was 62.1%.


## Introduction

We chose to use NLP to predict essay grades. As an English and Computer Science major, I (Naomi) am constantly interested in finding a meaningful intersection between these two fields. This is particularly because English is is the study of art, which is often not calculated but fluid, while Computer Science revolves around algorithmic concreteness. I think writing itself is an art and the evaluation of it is often entirely subjective. Trying to reframe this evaluation as algorithmic unites two seemingly disparate processes and allows us insights into why an essay succeeds. The factors that go into writing a good essay, to me, are not formulaic. There is no algorithm to writing a A+ paper because each topic requires something different and the evaluation of these paper comes from humans, who are inherently biased. This task seemed challenging and interesting because we attempted are attempting to simulate a human's process of evaluating and reasoning about a paper. This problem can be tackled, not necessarily solved, using NLP because MaxEnt will take information about the paper and determine how this information relates to the paper's grade. Rather than reading the paper itself, the information is abstracted as feature vectors.

#### Please see paper.pdf for the full writeup.
