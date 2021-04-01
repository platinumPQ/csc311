from utils import *

import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    user_id, question_id, is_correct = data["user_id"], data["question_id"], data["is_correct"]
    for k in range(len(user_id)):
        theta_i = theta[user_id[k]]
        beta_j = beta[question_id[k]]
        log_lklihood += is_correct[k]*(theta_i-beta_j)+beta_j-logsumexp([beta_j, theta_i])
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    user_id, question_id, is_correct = data["user_id"], data["question_id"], data["is_correct"]
    # initiate the g_theta and g_beta
    g_theta = np.zeros(theta.shape)
    g_beta = np.zeros(beta.shape)
    for k in range(len(user_id)):
        theta_i = theta[user_id[k]]
        beta_j = beta[question_id[k]]
        g_theta[user_id[k]] += sigmoid(theta_i-beta_j) - is_correct[k]
        g_beta[question_id[k]] += is_correct[k] - sigmoid(theta_i-beta_j)
    # gradient descent update to find the min of neg-log likelihood
    theta -= lr * g_theta
    beta -= lr * g_beta
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.random.rand(542)
    beta = np.random.rand(1774)

    train_lld = []
    val_lld = []
    train_acc_lst = []
    val_acc_lst = []

    for i in range(iterations):
        neg_lld_train = neg_log_likelihood(data, theta=theta, beta=beta)
        neg_lld_val = neg_log_likelihood(val_data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        train_acc = evaluate(data=data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        train_acc_lst.append(train_acc)
        train_lld.append(-neg_lld_train)
        val_lld.append(-neg_lld_val)
        # print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, train_acc_lst, val_acc_lst, train_lld, val_lld


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    # a1 = irt(train_data, val_data, 0.1, 50)
    # a2 = irt(train_data, val_data, 0.01, 50)
    # a3 = irt(train_data, val_data, 0.01, 100)
    a4 = irt(train_data, val_data, 0.015, 50)

    # print(f"For learning rate = 0.1, iteration = 50, the validation accuracy is\n{a1[3]}")
    # print(f"For learning rate = 0.01, iteration = 50, the validation accuracy is\n{a2[3]}")
    # print(f"For learning rate = 0.01, iteration = 100, the validation accuracy is\n{a3[3]}")
    print(f"For learning rate = 0.015, iteration = 50, the validation accuracy is\n{a4[3]}")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (c)
    theta_final = a4[0]
    beta_final = a4[1]
    val_acc = evaluate(val_data, theta_final, beta_final)
    test_acc = evaluate(test_data, theta_final, beta_final)
    print(f"With chosen learning rate = 0.015 and iteration = 50,\n"
          f"The final validation accuracy is {val_acc}\n"
          f"The final test accuracy is {test_acc}")
    plt.figure(1)
    plt.plot(list(range(50)), a4[4], label="train loglikelihhood")
    plt.plot(list(range(50)), a4[5], label="validation loglikelihhood")
    plt.xlabel("iteration")
    plt.ylabel("log likelihood")
    plt.legend()
    plt.figure(1)
    #####################################################################
    # Question 2(d)
    theta = np.sort(theta_final)
    plt.figure(2)
    for j in range(5):
        beta_j = beta_final[j]
        cij = sigmoid(theta - beta_j)
        plt.plot(theta, cij, label="Question #"+str(j))
    plt.title("Probability of Correct Response vs. Theta")
    plt.ylabel("Probability")
    plt.xlabel("Theta")
    plt.legend()
    plt.figure(2)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
