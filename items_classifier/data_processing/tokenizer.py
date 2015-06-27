# -*- coding: utf-8 -*-

import os

import nltk.downloader
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from items_classifier.configs.config import NLTK_DIR

nltk.data.path.append(NLTK_DIR)

def download_corpus():
    downloader = nltk.downloader.Downloader(download_dir=NLTK_DIR)
    downloader.download('wordnet', download_dir=NLTK_DIR)

if not os.path.exists(os.path.join(NLTK_DIR, 'corpora', 'wordnet')):
    download_corpus()


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]