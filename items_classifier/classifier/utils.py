from collections import defaultdict
import numpy
import sklearn
import warnings

CV_FOLDS_COUNT = 5
CV_TEST_SET_FRACTION = 0.3


def get_accuracy(classifier, X, y, n_iter, test_size, scoring='f1'):
    fold = sklearn.cross_validation.ShuffleSplit(len(X), n_iter=n_iter, test_size=test_size, random_state=0)
    features = classifier.vectorizer.transform(X)

    target = classifier.label_encoder.transform(y)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        scores = sklearn.cross_validation.cross_val_score(classifier.model, features, target, cv=fold, scoring=scoring)

    return scores.mean(), scores.std() * 2


def get_accuracy_for_each_class(classifier, X, y, n_iter=CV_FOLDS_COUNT, test_size=CV_TEST_SET_FRACTION):
    # sklearn does not allow to return multiple scores for each fold (i.e. score per class), so we have to use this hack
    labels_to_scores = defaultdict(list)

    def proxy_f1(ground_truth, predictions):
        for label in set(ground_truth):
            # encoding: 0 is a class that != label, 1 otherwise
            cur_ground_truth = [int(label == p) for p in ground_truth]
            cur_predictions = [int(label == p) for p in predictions]

            label_name = classifier.label_encoder.inverse_transform(label)
            f1_score = sklearn.metrics.f1_score(cur_ground_truth, cur_predictions, pos_label=1)
            labels_to_scores[label_name].append(f1_score)

        # emulate ordinary behavior
        return sklearn.metrics.f1_score(ground_truth, predictions, pos_label=None)

    common_result = get_accuracy(classifier, X, y, n_iter, test_size, scoring=sklearn.metrics.make_scorer(proxy_f1))

    for label, scores in labels_to_scores.items():
        scores = numpy.array(scores)
        labels_to_scores[label] = (scores.mean(), scores.std() * 2)

    return common_result, labels_to_scores