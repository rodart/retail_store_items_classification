import json

def filter_data(path):
    filtered_data = []
    with open(path) as data_file:
        for line in data_file:
            item = {}
            for field_name in ['title', 'categories', 'brand']:
                item[field_name] = json.loads(line).get(field_name)
            filtered_data.append(item)
    return filtered_data


def filter_corpora_by_each_class_size(corpora, lower_limit, upper_limit):
    return {label: corpora[label] for label in corpora if lower_limit < len(corpora[label]) < upper_limit}