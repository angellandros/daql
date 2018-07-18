from collections import defaultdict
import logging
import numpy as np
import random

from task2.BaselineRecommenderSystem import BaselineRecommenderSystem
from task3.AdvancedRecommenderSystem import AdvancedRecommenderSystem


def divide_the_data(file_name, k=5, dump=True, user_count=None):
    """
    >>> divided = divide_the_data('task2/example.csv', 5, False)
    >>> len(divided)
    5
    >>> [len(divided[i]['User1']) for i in range(5)]
    [2, 1, 1, 1, 1]
    """
    # load the data
    data_by_user = defaultdict(list)  # { user : [(item, rate, timestamp), ...], ... }
    with open(file_name, 'r') as f:
        for row in f:
            user, item, rate, timestamp = row.split(',')
            data_by_user[user].append((item, rate, timestamp))

    # handle user count
    logging.debug('user_count  input: ' + str(user_count))
    number_of_all_users = len(data_by_user)
    user_count = user_count or number_of_all_users  # handle user_count=None
    user_count = float(user_count)
    if 0.0 < user_count < 1.0:
        user_count = user_count * number_of_all_users
    user_count = int(user_count)
    logging.debug('user_count output: ' + str(user_count))

    # make k folds
    folds = []  # containing k subsets of the data, each of them having the same structure as data_by_user
    for i in range(k):
        folds.append(defaultdict(list))  # creating k maps of lists in the folds
    user_index = 0
    for user, ratings in data_by_user.items():
        if user_index >= user_count:
            logging.debug('breaking after %d user(s)' % user_index)
            break
        random.shuffle(ratings)
        for i in range(len(ratings)):
            fold_index = i % k
            folds[fold_index][user].append(ratings[i])
        user_index += 1

    if dump:
        # dump the data into files
        for fold_index in range(k):
            with open('ratings_fold_%d.csv' % fold_index, 'w') as f:
                for user, ratings in folds[fold_index].items():
                    for item, rate, timestamp in ratings:
                        f.write('%s,%s,%s,%s' % (user, item, rate, timestamp))  # timestamp contains newline character

    return folds


def metric_rmse(pred: map, echt: map):
    """
    >>> metric_rmse({'a': 3.0, 'b': 2.0}, {'a': 3.0, 'c': 1.0})
    0.0
    >>> metric_rmse({'a': 3.0, 'b': 2.0}, {'a': 2.0, 'c': 1.0})
    1.0
    """
    common_keys = set(pred.keys()).intersection(set(echt.keys()))
    return np.mean([(pred[key] - echt[key]) ** 2 for key in common_keys]) if len(common_keys) != 0 else 0


def metric_mae(pred: map, echt: map):
    """
    >>> metric_mae({'a': 3.0, 'b': 2.0}, {'a': 3.0, 'c': 1.0})
    0.0
    >>> metric_mae({'a': 3.0, 'b': 2.0}, {'a': 2.0, 'c': 1.0})
    1.0
    """
    common_keys = set(pred.keys()).intersection(set(echt.keys()))
    return np.mean([abs(pred[key] - echt[key]) for key in common_keys]) if len(common_keys) != 0 else 0


def metric_precision_at_k(pred: list, echt: list, k: int):
    """
    >>> metric_precision_at_k([('a', 3.0), ('b', 2.0)], [('a', 4.0), ('c', 3.0)], 1)
    1.0
    >>> metric_precision_at_k([('a', 3.0), ('b', 2.0)], [('a', 4.0), ('c', 3.0)], 2)
    0.5
    >>> metric_precision_at_k([('a', 3.0), ('b', 2.0)], [('a', 4.0), ('c', 3.0), ('b', 1.0)], 2)
    0.5
    """
    echt_keys = [key for key, _ in echt[:k]]
    return sum([1.0 for key, _ in pred[:k] if key in echt_keys]) / float(k)


def metric_recall_at_k(pred: list, echt: list, k: int):
    """
    >>> metric_recall_at_k([('a', 3.0), ('b', 2.0)], [('a', 4.0), ('c', 3.0)], 1)
    0.5
    >>> metric_recall_at_k([('a', 3.0), ('b', 2.0)], [('a', 4.0), ('c', 3.0)], 2)
    0.5
    >>> metric_recall_at_k([('a', 3.0), ('b', 2.0)], [('a', 4.0), ('c', 3.0), ('b', 1.0)], 2)
    0.6666666666666666
    """
    echt_keys = [key for key, _ in echt]
    return sum([1.0 for key, _ in pred[:k] if key in echt_keys]) / len(echt)


def metric_mrr_at_k(pred: list, echt: list, k: int):
    """
    >>> metric_mrr_at_k([('a', 3.0), ('b', 2.0)], [('a', 4.0), ('c', 3.0)], 1)
    1.0
    >>> metric_mrr_at_k([('a', 3.0), ('b', 2.0)], [('a', 4.0), ('c', 3.0)], 2)
    0.5
    >>> metric_mrr_at_k([('a', 3.0), ('b', 2.0)], [('a', 4.0), ('c', 3.0), ('b', 1.0)], 3)
    0.5
    """
    echt_keys = [key for key, _ in echt[:k]]
    pred_keys = [key for key, _ in pred]
    return sum([1.0 / (pred_keys.index(key) + 1.0) for key in echt_keys if key in pred_keys]) / float(k)


def test_once(train_sets, loaded_test_set, k=10):
    engine = AdvancedRecommenderSystem()
    for train_set in train_sets:
        engine.loadRatings(train_set)
        logging.debug('engine size: %d' % sum(len(val) for _, val in engine.ratingsIndex.items()))

    test_results = defaultdict(list)

    for user, ratings in loaded_test_set.items():
        pred_list = engine.predictTopKRecommendations(user, len(ratings))
                                                        # subset_fuzziness=0.01, similarity_threshold=0.2)
        logging.debug(('results for user %s: ' % user) + str(pred_list))
        # compute errors and other metrics
        pred_dict = {item: rate for item, rate in pred_list if type(rate) == float}
        echt_dict = {item: float(rate) for item, rate, _ in ratings}
        echt_list = sorted(echt_dict.items(), key=lambda x: -x[1])
        common_keys = set(pred_dict.keys()).intersection(set(echt_dict.keys()))
        logging.debug('number of correct recommendations for user %s: %d/%d' % (user, len(common_keys), len(ratings)))
        test_results['rmse'].append(metric_rmse(pred_dict, echt_dict))
        test_results['mae'].append(metric_rmse(pred_dict, echt_dict))
        test_results['precision@k'].append(metric_precision_at_k(pred_list, echt_list, k))
        test_results['recall@k'].append(metric_recall_at_k(pred_list, echt_list, k))
        test_results['mrr@k'].append(metric_mrr_at_k(pred_list, echt_list, k))

    logging.debug(str(test_results))

    return test_results, {key: np.mean(value) for key, value in test_results.items()}


def test_5_times(folds, names, k=10):
    test_results = defaultdict(list)
    for i in range(5):
        _, mean_result = test_once([names[j] for j in range(5) if i != j], folds[i], k=k)
        for key, value in mean_result.items():
            test_results[key].append(value)
    return {key: np.mean(value) for key, value in test_results.items()}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-set', help='the csv file of ratings', default='ratings.csv')
    parser.add_argument('-u', '--user-count', help='the number or ratio of users to use in the test, e.g. 20, 0.25',
                        default=None)
    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()
    folds = divide_the_data(args.data_set, user_count=args.user_count)
    names = ['ratings_fold_%d.csv' % i for i in range(5)]
    results = test_5_times(folds, names, k=10)
    print(results)
