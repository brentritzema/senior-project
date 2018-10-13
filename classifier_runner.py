import sys
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Perceptron
from sklearn import metrics
# load the data, which will be split into train and test sets

# make a vectorizers which 

full_training_dataset = pd.read_csv("/home/bjr33/workspace/cs/data/stance/coding/auto_trainset_tok.csv")

training_dataset_x = full_training_dataset['tweet_t']
training_dataset_y = full_training_dataset['stance']

print(training_dataset_x.size)
print(training_dataset_y.size)

full_testing_dataset = pd.read_csv("/home/bjr33/workspace/cs/data/stance/coding/gold_20180514_majority_fixed_tok.csv")

testing_dataset_x = full_testing_dataset['tweet_t']
testing_dataset_y = full_testing_dataset['stance']

vectorizer = TfidfVectorizer(ngram_range=(1,3), analyzer='char', use_idf=False)

clf = Pipeline([
    ('vec', vectorizer),
    ('clf',Perceptron(tol=1e-3))
])

clf.fit(training_dataset_x, training_dataset_y)

predictions = clf.predict(testing_dataset_x)

print(metrics.classification_report(testing_dataset_y, predictions))

cm = metrics.confusion_matrix(testing_dataset_y, predictions)
print(cm)