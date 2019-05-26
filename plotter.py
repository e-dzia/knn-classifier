import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_file(filename):
    dataset = pandas.read_csv(filename)
    return dataset


def plot_tests(df, test, filename):
    test_name = test[0]
    test_groupby = test[1]
    test_x = test[2]

    scores = ['acc', 'pr', 'rec', 'f1']

    df = df.query("tests == '{}'".format(test_name))

    for score in scores:
        fig, ax = plt.subplots()
        for key, grp in df.groupby([test_groupby]):
            ax = grp.plot(ax=ax, kind='line', x=test_x, y=score, label=key, style='.-',
                          title='Miara {}, testy {}'.format(score, test_name))
            #ax.set_ylim(0, 1)
        #plt.show()
        plt.savefig('plots/{}-{}-{}.png'.format(filename, test_name, score))


def plot_all_tests(filename):
    df = load_file(filename)
    tests = [('crossvalidation', 'stratified', 'splits'), ('metric', 'metric', 'k'), ('weight', 'weight', 'k')]
    filename_base = filename.split('/')
    filename_base = filename_base[1].split('.csv')
    for test in tests:
        plot_tests(df, test, filename_base[0])


if __name__ == "__main__":
    plot_all_tests('results/res-2019-05-12_14.02.07.012091-iris-1.csv')
