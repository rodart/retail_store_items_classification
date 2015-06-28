# -*- coding: UTF-8 -*-

import sklearn.svm
import sklearn.feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy

from items_classifier.utils.log import get_logger
from items_classifier.classifier.utils import get_accuracy_for_each_class
from items_classifier.utils.files import get_model_path, read_csv, get_cached
from items_classifier.data_processing.tokenizer import LemmaTokenizer

TOKEN_PATTERN = r'(?u)(?:\w\w+)'
ZERO_THRESHOLD = 0


class Classifier(object):
    def __init__(self, classifier_algo, vectorizer):
        self._model = classifier_algo
        self._vectorizer = vectorizer
        self._label_encoder = sklearn.preprocessing.LabelEncoder()

    @property
    def model(self):
        return self._model

    @property
    def label_encoder(self):
        return self._label_encoder

    @property
    def vectorizer(self):
        return self._vectorizer

    def fit(self, X_train, y_train):
        features = self._vectorizer.fit_transform(X_train)
        self._label_encoder.fit(y_train)
        self._model.fit(features, self._label_encoder.transform(y_train))

    def predict(self, sample):
        feature_vector = self._vectorizer.transform(sample)
        result = self._model.predict(feature_vector)
        return self._label_encoder.inverse_transform(result)[0]

    def decision_function(self, sample):
        feature_vector = self._vectorizer.transform(sample)

        if hasattr(self._model, 'decision_function') and callable(getattr(self._model, 'decision_function')):
            predicted_scores = self._model.decision_function(feature_vector)[0]
        elif hasattr(self._model, 'predict_proba') and callable(getattr(self._model, 'predict_proba')):
            predicted_scores = self._model.predict_proba(feature_vector)[0]
        else:
            raise Exception('No suitable method for ' + self._model.__class__.__name__)

        labels = self._label_encoder.inverse_transform(self._model.classes_)

        if len(labels) > 2:
            result = {labels[i]: predicted_scores[i] for i in xrange(len(labels))}
        else:
            predicted_label = labels[0] if predicted_scores <= 0 else labels[1]
            predicted_score = predicted_scores
            result = {predicted_label: predicted_score}

        return result

    def weighted_predict(self, sample):
        decision_func = self.decision_function(sample)
        most_confident = max(decision_func.keys(), key=lambda x: decision_func[x])
        return {most_confident: decision_func[most_confident]}


class TextClassifier(object):
    def __init__(self, corpus_path, threshold=None, crossvalidate=True):
        self._logger = get_logger(self.__module__ + '.' + self.__class__.__name__)
        self._threshold = float('-inf') if threshold is None else threshold

        builder = self._get_classifier_builder(
            train_corpus_path=corpus_path,
            crossvalidate=crossvalidate,
            logger=self._logger
        )

        self._classifier = get_cached(builder, get_model_path(corpus_path))

    @classmethod
    def _get_classifier_builder(cls, train_corpus_path, crossvalidate, logger):
        def _builder():
            train_corpus = cls.get_corpus(train_corpus_path)

            vectorizer = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word',
                                         token_pattern=TOKEN_PATTERN, ngram_range=(1, 3), use_idf=1, smooth_idf=1,
                                         sublinear_tf=1, stop_words='english', tokenizer=LemmaTokenizer())

            classifier_algo = sklearn.svm.LinearSVC()
            classifier = Classifier(classifier_algo, vectorizer)

            X_train, y_train = cls._get_training_matricies(train_corpus)
            classifier.fit(X_train, y_train)

            if crossvalidate:
                common_result, each_class_result = get_accuracy_for_each_class(classifier, X_train, y_train)
                logger.info(u'total accuracy: %0.2f ± %0.2f' % common_result)
                for label, result in each_class_result.items():
                    logger.info(label + u' accuracy: %0.2f ± %0.2f' % result)
            return classifier

        return _builder

    @staticmethod
    def _get_training_matricies(corpus):
        X, y = [], []
        for k, v in corpus.items():
            for x in v:
                X.append(x)
                y.append(k)
        y = numpy.array(y)
        return X, y

    @property
    def classifier(self):
        return self._classifier

    @staticmethod
    def get_corpus(file_name):
        raw_data_lines = read_csv(file_path=file_name, preserve_header=True)

        header = [x.strip() for x in raw_data_lines[0]]
        columns = range(len(header))
        raw_data = [[] for _ in columns]

        for line in raw_data_lines[1:]:
            for i in columns:
                if line[i]:
                    raw_data[i].append(line[i].strip())

        return {header[i]: raw_data[i] for i in columns}

    def predict(self, prepared_phrase):
        result = self._classifier.weighted_predict([prepared_phrase])
        label = next(iter(result.keys()))
        result = set() if result[label] <= self._threshold else set([label])

        return result

    def weighted_predict(self, text):
        result = self._classifier.weighted_predict([text])
        label = next(iter(result.keys()))
        if result[label] <= self._threshold:
            result.pop(label)

        return result

    def decision_function(self, text):
        result = self._classifier.decision_function([text])
        return {label: conf for label, conf in result.items() if conf > self._threshold}