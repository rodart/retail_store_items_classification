from items_classifier.data_processing.data_filtering import filter_data, filter_corpora_by_each_class_size
from items_classifier.utils.files import save_as_json, read_as_json, save_json_as_csv
from items_classifier.data_processing.category_filtering import get_all_level_categories, get_n_level_categories_corpora
from items_classifier.classifier.text_classifier import TextClassifier

UNFILTERED_DATA_PATH = 'data/meta_Electronics.json'
FILTERED_DATA_PATH = 'data/filtered_meta_Electronics.json'
CATEGORIES_CORPORA_PATH = 'data/categories_corpora.csv'


def filter_raw_data():
    """
    remove useless fields
    """
    data = filter_data(UNFILTERED_DATA_PATH)
    save_as_json(data, FILTERED_DATA_PATH)


def print_categories():
    """
    print all items categories in filtered dataset
    """
    data = read_as_json(FILTERED_DATA_PATH)
    categories = get_all_level_categories(data)
    categories = sorted(categories)
    for category in categories:
        print category


def make_train_set():
    """
    make train set from filtered data with specified categories level and corpus size limits
    """
    data = read_as_json(FILTERED_DATA_PATH)
    corpora = get_n_level_categories_corpora(data, level=2)
    corpora = filter_corpora_by_each_class_size(corpora, lower_limit=200, upper_limit=1000)
    save_json_as_csv(corpora, CATEGORIES_CORPORA_PATH)


if __name__ == '__main__':
    # filter_raw_data()
    # print_categories()
    # make_train_set()

    classifier = TextClassifier(corpus_path=CATEGORIES_CORPORA_PATH, crossvalidate=True)
    title = 'NETGEAR NeoTV Max Streaming Player (NTV300SL)'

    print classifier.predict(title)
    print classifier.weighted_predict(title)
    print classifier.decision_function(title)