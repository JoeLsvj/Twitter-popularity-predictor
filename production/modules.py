# packages for importing and managing tweets
import tweepy
import snscrape.modules.twitter as sntwitter

# basics of numerics, data structures, statistics
import pandas as pd
import numpy as np
from time import time
import os
import sys
from pprint import pprint

# importing other packages for text preprocessing
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import regex
import string
import random
from gensim.models import Word2Vec
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
#from textblob import TextBlob
import unicodedata
import emoji  ## pay attention to the version... With version 0.6.0 works
              ## with version 1.7 the attribute UNICODE_EMOJI does not exists -> EMOJI_DATA.


# plotting and graphics
import matplotlib.pyplot as plt

# The Big package for Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
# import the vectorizers from skearn:
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# import a learning technique for the next step:
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import MultinomialNB   #classifier
# Import also a support vector machine (SVM)
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.dummy import DummyRegressor
# Importing functions for feature processing and analysis
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MaxAbsScaler  # for the sparse objects
# Importing functions for accessing the models (from effectiveness axis)
from sklearn.model_selection import *
from sklearn.model_selection import validation_curve
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
