import csv
import datetime
import random
import warnings
import pandas
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import naive_bayes, model_selection, metrics, preprocessing
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

from imad_4.printer import show_data


def unpack_data(filename):
    dataset = pandas.read_csv(filename, header=None)

    if filename == 'files/iris.csv':
        dataset.columns = ["petalLength", "petalWidth", "sepalLength", "sepalWidth", "class"]

    if filename == 'files/glass.csv':
        dataset.columns = ["id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "class"]
        dataset = dataset.set_index('id')

    if filename == 'files/pima-diabetes.csv':
        dataset.columns = ["NumTimesPrg", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc",
                           "Age", "class"]

    if filename == 'files/wine.csv':
        dataset.columns = ["class", "Alcohol", "MalicAcid", "Ash", "AlcalinityOfAsh", "Magnesium", "TotalPhenols",
                           "Flavanoids", "NonflavanoidPhenols", "Proanthocyanins", "ColorIntensivity", "Hue",
                           "OD280/OD315", "Proline"]

    if filename == 'files/customers.csv':
        dataset.columns = ["Channel", "Region", "Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]

    return dataset


def preprocess_data(X):
    # normalize [0,1]
    columns = X.columns
    X = pandas.DataFrame(preprocessing.normalize(X), columns=columns)

    return X


def split_data(dataset):
    # split into train and test sets
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42, stratify=dataset['class'])

    # extract labels
    train_set_labels = train_set["class"].copy()
    train_set = train_set.drop("class", axis=1)

    test_set_labels = test_set["class"].copy()
    test_set = test_set.drop("class", axis=1)

    return train_set, train_set_labels, test_set, test_set_labels


def extract_labels(dataset):
    # extract labels
    dataset_labels = dataset["class"].copy()
    dataset = dataset.drop("class", axis=1)

    return dataset, dataset_labels


def cross_validation(X, y, kfold, model):
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    for train, test in kfold.split(X, y):
        train = train.tolist()
        test = test.tolist()
        params = model.get_params()
        t = type(model)
        model = t(**params)
        model.fit(X.iloc[train], y.iloc[train])
        labels_predicted = model.predict(X.iloc[test])
        labels_true = y.iloc[test]
        accuracy, precision, recall, f1 = evaluate(labels_true, labels_predicted)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    accuracy = np.mean(accuracies)
    precision = np.mean(precisions)
    recall = np.mean(recalls)
    f1 = np.mean(f1s)
    return accuracy, precision, recall, f1


def evaluate(labels_true, labels_predicted):
    accuracy = metrics.accuracy_score(y_true=labels_true, y_pred=labels_predicted)
    precision = metrics.precision_score(y_true=labels_true, y_pred=labels_predicted, average='macro')
    recall = metrics.recall_score(y_true=labels_true, y_pred=labels_predicted, average='macro')
    f1 = metrics.f1_score(y_true=labels_true, y_pred=labels_predicted, average='macro')

    return accuracy, precision, recall, f1


def square_distance_weights(distances):
    weights = np.array([(1/(distance**2)) for distance in distances])
    return weights


def main_single(filename, show_mode, n_neighbors=3, weights='uniform', p=2, splits=10, stratified=True):
    # unpack the data from .csv
    dataset = unpack_data(filename)

    # choose the model
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p, n_jobs=-1)

    # split the data
    dataset, dataset_labels = extract_labels(dataset)
    dataset = preprocess_data(dataset)

    if stratified:
        kfold = model_selection.StratifiedKFold(n_splits=splits, shuffle=True, random_state=random.randint(0, 10000))
    else:
        kfold = model_selection.KFold(n_splits=splits, shuffle=True, random_state=random.randint(0, 10000))

    accuracy, precision, recall, f1 = cross_validation(dataset, dataset_labels, kfold, model)

    if show_mode:
        print("Accuracy: {}, Precision: {}, Recall: {}, F1: {}".format(accuracy, precision, recall, f1))

    return accuracy, precision, recall, f1
