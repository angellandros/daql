from collections import defaultdict
import logging
import numpy as np
import random

from task2.BaselineRecommenderSystem import BaselineRecommenderSystem


def divide_the_data(file_name, k=5, dump=True, user_count=None):
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


def test_once(train_sets, loaded_test_set, k=10):
    engine = BaselineRecommenderSystem()
    for train_set in train_sets:
        engine.loadRatings(train_set)
        logging.debug('engine size: %d' % sum(len(val) for _, val in engine.ratingsIndex.items()))

    test_results = defaultdict(list)

    for user, ratings in loaded_test_set.items():
        results_all = engine.predictTopKRecommendations(user, len(ratings),
                                                        subset_fuzziness=0.01, similarity_threshold=0.2)
        logging.debug(('results for user %s: ' % user) + str(results_all))
        # compute RMSE (root-mean-squre error)
        pred = {item: rate for item, rate in results_all if type(rate) == float}
        echt = {item: float(rate) for item, rate, _ in ratings}
        common_keys = set(pred.keys()).intersection(set(echt.keys()))
        logging.debug('number of correct recommendations for user %s: %d/%d' % (user, len(common_keys), len(ratings)))
        rmse = np.mean([(pred[key] - echt[key]) ** 2 for key in common_keys]) if len(common_keys) != 0 else 0
        test_results['rmse'].append(rmse)

    print(test_results)

    return test_results


def test_k_times(folds, names, k=10):
    return test_once(names[1:], folds[0], k=k)


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
    test_k_times(folds, names, k=10)
