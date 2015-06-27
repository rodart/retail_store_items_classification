try:
    import cPickle as pickle
except ImportError:  # pragma: no cover
    import pickle

import json
import csv
import os

from items_classifier.utils.log import get_logger
from items_classifier.configs.config import MODELS_DIR

_logger = get_logger(__name__)

def save_as_json(data, path):
    with open(path, 'w') as outfile:
        json.dump(data, outfile)


def read_as_json(path):
    with open(path) as data_file:
        data = json.load(data_file)
    return data


def save_json_as_csv(data, path):
    output = csv.writer(open(path, 'wb+'))
    output.writerow(data.keys())  # header row

    max_size = max(len(data[label]) for label in data)
    prepared_data = []
    for i in xrange(max_size):
        row = []
        for label in data:
            if len(data[label]) > i:
                row.append(data[label][i])
            else:
                row.append(None)
        prepared_data.append(row)

    output.writerows(prepared_data)


def read_csv(file_path, preserve_header=False):
    with open(file_path) as f:
        reader = csv.reader(f)
        data = [line for line in reader]
        if not preserve_header:
            data = data[1:]
    data = [[unicode(x, 'utf-8') for x in line] for line in data]
    return data


def get_model_path(file_path):
    return os.path.join(MODELS_DIR, '%s.bin' % file_path.strip(os.sep))


def get_cached(factory, cache_file_name):
    """
    Loads cache if exists, otherwise calls factory and stores the results in the specified cache file
    :param factory:
    :param cache_file_name:
    :return:
    """
    filename = cache_file_name.encode('UTF-8')
    if os.path.exists(filename):
        cached = _deserialize(filename)
        _logger.info(u'Loaded {}'.format(cache_file_name))
        return cached

    _logger.info(u'Creating {}'.format(cache_file_name))
    item = factory()
    with _ensure_file(filename, 'wb') as f:
        pickle.dump(item, f)
    _logger.info(u'Created cache {}'.format(cache_file_name))
    return item


def _deserialize(filename):
    with open(filename, 'rb') as f:
        item = pickle.load(f)
    return item


def _ensure_file(file_name, mode='wt'):
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    return open(file_name, mode)