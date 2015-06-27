CATEGORY_STR = 'Electronics'

def get_all_level_categories(data):
    all_level_categories = set()
    for item in data:
        for category_list in item['categories']:
            if category_list[0] == CATEGORY_STR:
                all_level_categories.add('/'.join(category_list))

    return all_level_categories


def get_n_level_categories_corpora(data, level):
    corpora = {}
    for item in data:
        for category_list in item['categories']:
            if len(category_list) <= level or category_list[0] != CATEGORY_STR:
                continue

            label = '/'.join(category_list[0:level+1])
            if label not in corpora:
                corpora[label] = set()
            corpora[label].add(item['title'])

    for label in corpora:
        corpora[label] = list(corpora[label])

    return corpora