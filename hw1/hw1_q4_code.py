import numpy as np
import matplotlib.pyplot as plt

from np.random import permutation
from matplotlib import interactive


def extract(data):
    """Return X and t in the input data in dictionary data type"""
    X = data['X']
    t = data['t']
    return X, t


def combine(X, t):
    """Combine the data in the desired data type"""
    return {'X': X, 't': t}


def shuffle_data(data):
    """Return randomly permuted version of the data """
    X, t = extract(data)
    perm = permutation(len(X))
    # shuffle each X with its corresponding t at the same time
    X_shuffled = X[perm]
    t_shuffled = t[perm]
    return combine(X_shuffled, t_shuffled)


def split_data(data, num_folds, fold):
    """Split the data into data in selected fold block and the remaining data. Note that the num_folds should be exact
     division of the row length of X or t"""
    X, t = extract(data)

    # Split the X matrix into a list with num_folds blocks
    X_list = np.vsplit(X, num_folds)
    data_fold_X = X_list[fold-1]
    X_list.pop(fold-1)
    data_rest_X = np.vstack(X_list)

    # split the t vector into a list with num_folds blocks
    t_list = np.split(t, num_folds)
    data_fold_t = t_list[fold-1]
    t_list.pop(fold - 1)
    data_rest_t = np.concatenate(t_list)

    data_fold, data_rest = combine(data_fold_X, data_fold_t), combine(data_rest_X, data_rest_t)
    return data_fold, data_rest


def train_model(data, lambd):
    """Return the coefficient of the ridge expression with penalty lambd"""
    X, t = extract(data)
    xtrans = np.transpose(X)
    xtransx = np.dot(xtrans, X)
    lambdNI = np.identity(X.shape[1])*len(X)*lambd
    xtranst = np.dot(xtrans, t)
    inv = np.linalg.inv(xtransx+lambdNI)
    w = np.dot(inv, xtranst)
    return w


def predict(data, model):
    """Returns the prediction based on the data and model"""
    X, t = extract(data)
    return np.dot(X,model)


def loss(data, model):
    """Return the average squared error loss based on model"""
    X, t = extract(data)
    err = np.dot(X, model) - t
    err_trans = np.transpose(err)
    err_avg = np.dot(err_trans, err) / (2*len(X))
    return err_avg


def cross_validation(data, num_folds, lambd_seq):
    """Return the cross validation error vector across all lambda"""
    cv_error = []
    data = shuffle_data(data)
    for lambd in lambd_seq:
        cv_loss_lmd = 0
        for fold in range(1, num_folds+1):
            val_cv, train_cv = split_data(data, num_folds, fold)
            model = train_model(train_cv, lambd)
            cv_loss_lmd += loss(val_cv, model)
        cv_error.append(cv_loss_lmd/num_folds)
    return np.array(cv_error)


if __name__ == '__main__':
    interactive(False)

    trainpath_x = 'D:\AAAAAA YUPEIQING\大四\csc311\data\data_train_X.csv'
    trainpath_y = 'D:\AAAAAA YUPEIQING\大四\csc311\data\data_train_y.csv'
    testpath_x = 'D:\AAAAAA YUPEIQING\大四\csc311\data\data_test_X.csv'
    testpath_y = 'D:\AAAAAA YUPEIQING\大四\csc311\data\data_test_y.csv'

    data_train = {'X': np.genfromtxt(trainpath_x, delimiter=','), 't': np.genfromtxt(trainpath_y, delimiter=',')}
    data_test = {'X': np.genfromtxt(testpath_x, delimiter=','), 't': np.genfromtxt(testpath_y, delimiter=',')}

    # Q4(c) -- report the training and test error corresponding to each lambda
    train_err, test_err = [], []
    lambd_sq = np.linspace(0.00005, 0.005, 50)
    train_shuffled = shuffle_data(data_train)
    test_shuffled = shuffle_data(data_test)
    for lbd in lambd_sq:
        w = train_model(train_shuffled, lbd)
        train_err.append(loss(train_shuffled, w))
        test_err.append(loss(test_shuffled, w))

    # Q4(d) -- get the cross validation error for 5-fold and 10-fold
    error5 = cross_validation(train_shuffled, 5, lambd_sq)
    error10 = cross_validation(train_shuffled, 10, lambd_sq)

    small_err5, small_err10 = error5.min(), error10.min()
    i_err5, i_err10 = 0, 0
    for i in range(0, len(lambd_sq)):
        if error5[i] == small_err5:
            i_err5 = i
        if error10[i] == small_err10:
            i_err10 = i
    print('value of λ proposed by 5-fold cv:' + str(lambd_sq[i_err5]))
    print('value of λ proposed by 10-fold cv:' + str(lambd_sq[i_err10]))

    # plot the training and validation accuracy for each k
    plt.plot(lambd_sq, train_err, label='training error')
    plt.plot(lambd_sq, test_err, label='test error')
    plt.plot(lambd_sq, error5, label='5-fold')
    plt.plot(lambd_sq, error10, label='10-fold')
    plt.xlabel('lambda - penalty parameter')
    plt.ylabel('error rate')
    plt.legend()
    plt.show()

