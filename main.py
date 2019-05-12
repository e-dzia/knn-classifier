import datetime
import random
import warnings

from imad_4.plotter import plot_all_tests
from imad_4.utils import main_single, square_distance_weights
from imad_4.tests import main_tests


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=Warning)
    random.seed(datetime.datetime.now())
    show_mode = True
    filenames = ['iris.csv', 'pima-diabetes.csv', 'glass.csv', 'wine.csv']
    k_neighbors = [1, 2, 3, 4, 5, 7, 10, 15, 20]
    metrics = [1, 2]  # 1 - manhattan, 2 - euclidean
    weights = ['uniform', 'distance', square_distance_weights]
    splits_sizes = [2, 3, 5, 10]
    stratified = [False, True]

    filename = filenames[0]
    k = k_neighbors[2]
    p = metrics[1]
    weight = weights[2]
    splits = 10

    #main_single('files/' + filename, show_mode, n_neighbors=k, weights=weight, p=p, splits=10, stratified=True)
    outfiles = main_tests(1, filenames[1:], k_neighbors, weights, metrics, splits_sizes, stratified)

    for filename in outfiles:
        plot_all_tests(filename)
