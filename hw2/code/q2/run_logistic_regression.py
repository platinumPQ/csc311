from q2.check_grad import check_grad
from q2.utils import *
from q2.logistic import *

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def best_weights(train_inputs, train_targets, num_iterations, hyperparameters):
    """ This is a helper function that returns the weights for evaluation of test dataset with given iteration
    of the training dataset
    """
    N, M = train_inputs.shape
    weights = np.zeros((M + 1, 1))
    for t in range(num_iterations):
        # train the model with the training data
        f, df, y_train = logistic(weights, train_inputs, train_targets, hyperparameters)
        weights = weights - hyperparameters['learning_rate']*df
    return weights


def run_logistic_regression():
    # train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.01,
        "weight_regularization": 0.,
        "num_iterations": 400
    }
    weights = np.zeros((M+1, 1))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    # initialize the empty list
    lce_train, lfrac_correct_train = [], []
    lce_val, lfrac_correct_val = [], []

    tlist = np.arange(hyperparameters['num_iterations']) + 1

    for t in range(hyperparameters["num_iterations"]):
        # train the model with the training data
        f, df, y_train = logistic(weights, train_inputs, train_targets, hyperparameters)
        ce_train, frac_correct_train = evaluate(train_targets, y_train)
        lce_train.append(ce_train)
        lfrac_correct_train.append(frac_correct_train)
        # update the weight by gradient descent rule
        weights = weights - hyperparameters['learning_rate']*df

        # evaluate the prediction for validation data
        y_val = logistic_predict(weights, valid_inputs)
        ce_val, frac_correct_val = evaluate(valid_targets, y_val)
        lce_val.append(ce_val)
        lfrac_correct_val.append(frac_correct_val)

    # determine the largest iteration to get the smallest ce or largest classification rate
    t_train = max(lce_train.index(min(lce_train)), lfrac_correct_train.index(max(lfrac_correct_train)))
    best_ce_train = lce_train[t_train]
    best_frac_train = lfrac_correct_train[t_train]

    t_val = max(lce_val.index(min(lce_val)), lfrac_correct_val.index(max(lfrac_correct_val)))
    best_ce_val = lce_train[t_val]
    best_frac_val = lfrac_correct_val[t_val]

    # calculate the best weight
    best_w = best_weights(train_inputs, train_targets, t_train, hyperparameters)

    # load test data and evaluate the prediction
    test_inputs, test_targets = load_test()
    y_test = logistic_predict(best_w, test_inputs)
    ce_test, frac_correct_test = evaluate(test_targets, y_test)

    # plot the cross entropy of training and valid data
    plt.figure(10)
    plt.plot(tlist, lce_train, label='training CE')
    plt.plot(tlist, lce_val, label='validation CE')
    plt.xlabel('number of iterations')
    plt.ylabel('cross entropy loss')
    plt.title('mnist_train with learning rate= ' + str(hyperparameters['learning_rate']))
    plt.legend()

    # plot the classification rate of training and valid data
    plt.figure(11)
    plt.plot(tlist, lfrac_correct_train, label='training classification rate')
    plt.plot(tlist, lfrac_correct_val, label='validation classification rate')
    plt.xlabel('number of iterations')
    plt.ylabel('classification rate')
    plt.title('mnist_train with learning rate= ' + str(hyperparameters['learning_rate']))
    plt.legend()
    plt.figure(10)
    plt.figure(11)

    # print the output
    locals()
    print(f"For hyperparameter settings: learning rate = {hyperparameters['learning_rate']}, best training iteration ={t_train},"
          f"best validation iteration={t_train}.\nThe final cross entropy for training data = {best_ce_train}, "
          f"final ce for validation data ={best_ce_val}, final ce for test data ={ce_test}.\n"
          f"The final classification rate for training data = {best_frac_train}, "
          f"final classification rate for validation data ={best_frac_val},"
          f"final classification rate for test data ={frac_correct_test}." .format(**locals()))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_pen_logistic_regression():
    train_inputs, train_targets = load_train()
    # train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Implement the function that automatically evaluates different     #
    # penalty and re-runs penalized logistic regression 5 times.        #
    #####################################################################
    # set the hyperparameters. Note that the num_iterations should be exactly the one with least cross entropy or
    # highest classification rate !!!
    hyperparameters = {
        "learning_rate": 0.1,
        "weight_regularization": 0.,
        "num_iterations": 2000,
    }
    # weights = np.zeros((M + 1, 1))
    test_inputs, test_targets = load_test()
    lbd_list = [0, 0.001, 0.01, 0.1, 1.0]

    tlist = np.arange(hyperparameters['num_iterations']) + 1
    fig_num = 0

    lbd_ce_train, lbd_ce_val, lbd_rate_train, lbd_rate_val = [], [], [], []
    lbd_ce_test, lbd_rate_test = [], []

    for lbd in lbd_list:
        hyperparameters['weight_regularization'] = lbd
        rerun_ce_train, rerun_rate_train = [], []
        rerun_ce_val, rerun_rate_val = [], []

        for i in range(5):
            # N, M = train_inputs.shape
            weights = np.zeros((M + 1, 1))
            lce_train, lce_val = [], []
            for t in range(hyperparameters['num_iterations']):
                # train the model with the training data
                f, df, y_train = logistic_pen(weights, train_inputs, train_targets, hyperparameters)
                weights = weights - hyperparameters['learning_rate'] * df
                ce_train, frac_correct_train = evaluate(train_targets, y_train)
                lce_train.append(ce_train)

                # evaluate the prediction for validation data
                y_val = logistic_predict(weights, valid_inputs)
                ce_val, frac_correct_val = evaluate(valid_targets, y_val)
                lce_val.append(ce_val)
            rerun_ce_train.append(ce_train)
            rerun_rate_train.append(frac_correct_train)

            rerun_ce_val.append(ce_val)
            rerun_rate_val.append(frac_correct_val)

        lbd_ce_train.append(sum(rerun_ce_train)/len(rerun_ce_train))
        lbd_rate_train.append(sum(rerun_rate_train)/len(rerun_rate_train))
        lbd_ce_val.append(sum(rerun_ce_val) / len(rerun_ce_val))
        lbd_rate_val.append(sum(rerun_rate_val) / len(rerun_rate_val))

        # evaluate the prediction for test data
        y_test = logistic_predict(weights, test_inputs)
        ce_test, acc_test = evaluate(test_targets, y_test)
        lbd_ce_test.append(ce_test)
        lbd_rate_test.append(acc_test)

        plt.figure(fig_num)
        plt.plot(tlist, lce_train, label='training')
        plt.plot(tlist, lce_val, label='validation')
        plt.xlabel('number of iterations')
        plt.ylabel('cross entropy loss')
        plt.title("Change of Cross Entropy of mnist_train_small for lambda = "+str(lbd_list[fig_num]))
        plt.legend()
        plt.figure(fig_num)
        fig_num += 1

    # print the output
    locals()
    print(f"For the lambda value in {lbd_list}\n"
          f"The corresponding averaged cross entropy of mnist_train on the training set is {lbd_ce_train}\n"
          f"The corresponding averaged cross entropy of mnist_train on the validation set is {lbd_ce_val}\n"
          f"The corresponding averaged classification rate of mnist_train on the training set is {lbd_rate_train}\n"
          f"The corresponding averaged classification rate of mnist_train on the validation set is "
          f"{lbd_rate_val}\nThe test cross entropy is {lbd_ce_test}, and the accuracy is {lbd_rate_test}".format(**locals()))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
    run_pen_logistic_regression()
