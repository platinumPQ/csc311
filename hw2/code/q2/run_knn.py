from q2.l2_distance import l2_distance
from q2.utils import *

import matplotlib.pyplot as plt
import numpy as np


def knn(k, train_data, train_labels, valid_data):
    """ Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples,
          M is the number of features per example.

    :param k: The number of neighbours to use for classification
    of a validation example.
    :param train_data: N_TRAIN x M array of training data.
    :param train_labels: N_TRAIN x 1 vector of training labels
    corresponding to the examples in train_data (must be binary).
    :param valid_data: N_VALID x M array of data to
    predict classes for validation data.
    :return: N_VALID x 1 vector of predicted labels for
    the validation data.
    """
    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:, :k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # Note this only works for binary labels:
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1, 1)

    return valid_labels


def run_knn():
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    #####################################################################
    # TODO:                                                             #
    # Implement a function that runs kNN for different values of k,     #
    # plots the classification rate on the validation set, and etc.     #
    #####################################################################
    # initialize the classification list
    acc_val, acc_test = [], []
    for k in [1, 3, 5, 7, 9]:
        knn_valid = knn(k, train_inputs, train_targets, valid_inputs)
        acc_val.append(((knn_valid == valid_targets).sum())/len(valid_targets))
    plt.figure(0)
    plt.plot([1, 3, 5, 7, 9], acc_val)
    plt.xlabel('k - the number of neighbours in KNN')
    plt.ylabel('classification rate')
    plt.title('classification rate on validation set')
    plt.figure(0)

    # check the test performance for each k
    for k in [1, 3, 5, 7, 9]:
        knn_test = knn(k, train_inputs, train_targets, test_inputs)
        acc_test.append(((knn_test == test_targets).sum())/len(test_targets))
    plt.figure(1)
    plt.plot([1, 3, 5, 7, 9], acc_test)
    plt.xlabel('k - the number of neighbours in KNN')
    plt.ylabel('classification rate')
    plt.title('classification rate on test set')
    plt.figure(1)

    # print the output
    locals()
    print(f"For the k value in [1, 3, 5, 7, 9]\n"
          f"The corresponding validation accuracy is {acc_val}\n"
          f"The corresponding test accuracy is {acc_test}".format(**locals()))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    run_knn()
