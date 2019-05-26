import csv
import datetime
import numpy as np

from imad_4.utils import main_single, square_distance_weights


def main_single_tests(num_of_tests, filename, show_mode=False, n_neighbors=3, weights='uniform', p=2, splits=10, stratified=True):
    accs = []
    prs = []
    recs = []
    f1s = []
    for _ in range(num_of_tests):
        acc, pr, rec, f1 = main_single(filename, show_mode, n_neighbors=n_neighbors, weights=weights, p=p,
                                       splits=splits, stratified=stratified)
        accs.append(acc)
        prs.append(pr)
        recs.append(rec)
        f1s.append(f1)

    acc = np.mean(accs)
    pr = np.mean(prs)
    rec = np.mean(recs)
    f1 = np.mean(f1s)
    var_acc = np.sqrt(np.var(accs))
    var_pr = np.sqrt(np.var(prs))
    var_rec = np.sqrt(np.var(recs))
    var_f1 = np.sqrt(np.var(f1s))

    return acc, pr, rec, f1, var_acc, var_pr, var_rec, var_f1


def main_tests(num_of_tests, filenames, n_neighbors, weights, metrics, splits_sizes, stratified, show_mode=False):
    start = datetime.datetime.now()
    print("time start: {}".format(start))
    outfiles = []

    for filename in filenames:
        file = 'results/res-{}-{}-{}.csv'.format(start, filename.split(".")[0], num_of_tests).\
            replace(' ', '_').replace(':', '.')
        outfiles.append(file)
        f = open(file, "w", newline='')
        writer = csv.writer(f)
        writer.writerow(['file', 'tests', 'stratified', 'splits', 'k', 'weight', 'metric',
                         'acc', 'pr', 'rec', 'f1', 'acc_std', 'pr_std', 'rec_std', 'f1_std'])

        k = 3
        weight = 'uniform'
        p = 2

        for stratify in stratified:
            for splits in splits_sizes:
                results = main_single_tests(num_of_tests, 'files/' + filename, False,
                                            n_neighbors=k, weights=weight, p=p,
                                            splits=splits, stratified=stratify)
                data = [filename, 'crossvalidation', 'strat' if stratify else 'not',
                        splits, k, weight if weight != square_distance_weights else 'square',
                        'euclidean' if p == 2 else 'manhattan']
                data.extend(results)
                writer.writerow(data)
                if show_mode:
                    print(data)

        stratify = True
        splits = 10

        for p in metrics:
            for k in n_neighbors:
                results = main_single_tests(num_of_tests, 'files/' + filename, False,
                                            n_neighbors=k, weights=weight, p=p,
                                            splits=splits, stratified=stratify)
                data = [filename, 'metric', 'strat' if stratify else 'not',
                        splits, k, weight if weight != square_distance_weights else 'square',
                        'euclidean' if p == 2 else 'manhattan']
                data.extend(results)
                writer.writerow(data)
                if show_mode:
                    print(data)

        for weight in weights:
            for k in n_neighbors:
                results = main_single_tests(num_of_tests, 'files/' + filename, False,
                                            n_neighbors=k, weights=weight, p=p,
                                            splits=splits, stratified=stratify)
                data = [filename, 'weight', 'strat' if stratify else 'not',
                        splits, k, weight if weight != square_distance_weights else 'square',
                        'euclidean' if p == 2 else 'manhattan']
                data.extend(results)
                writer.writerow(data)
                if show_mode:
                    print(data)

        end = datetime.datetime.now()
        print("done, time: {}, elapsed: {}".format(end, end - start))
        f.close()
    return outfiles
