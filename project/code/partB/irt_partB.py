from gender import *
from age import *
from item_response import *
from utils import *
import matplotlib.pyplot as plt


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
    """ Modification of the item_response.py. Here we use the student metadata and separate the
    train, test, validation dataset into male and female group, and small 7-13, medium 13-16,
    large 16-18 age group
        """
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # separate the train, validation, and test data by gender
    boy, girl = separate_gender()
    b_test, g_test = get_dic(boy, girl, test_data)
    b_train, g_train = get_dic(boy, girl, train_data)
    b_val, g_val = get_dic(boy, girl, val_data)

    # train each male and female datasets separately
    boy = irt(b_train, b_val, 0.015, 50)
    girl = irt(g_train, g_val, 0.015, 50)
    print(f"For learning rate = 0.015, iteration = 50, the train accuracy for boys is\n{boy[2]}")
    print(f"For learning rate = 0.015, iteration = 50, the train accuracy for girls is\n{girl[2]}")

    b_theta = boy[0]
    b_beta = boy[1]
    g_theta = girl[0]
    g_beta = girl[1]

    # obtain the test accuracy of male and female algorithm
    b_test_acc = evaluate(b_test, b_theta, b_beta)
    g_test_acc = evaluate(g_test, g_theta, g_beta)

    print(f"With chosen learning rate = 0.015 and iteration = 50,\n"
          f"The final test accuracy for boys is {b_test_acc}\n"
          f"The final test accuracy for girls is {g_test_acc}")

    # separate the train, validation, and test data by gender
    low, mid, high = separate_age()
    l_train, m_train, h_train = get_dict(low, mid, high, train_data)
    l_test, m_test, h_test = get_dict(low, mid, high, test_data)
    l_val, m_val, h_val = get_dict(low, mid, high, val_data)

    # train each age datasets separately
    low = irt(l_train, l_val, 0.015, 50)
    medium = irt(m_train, m_val, 0.015, 50)
    high = irt(h_train, h_val, 0.015, 50)
    print(f"For learning rate = 0.015, iteration = 50, the train accuracy for age 7-13 is\n{low[2]}")
    print(f"For learning rate = 0.015, iteration = 50, the train accuracy for age 13-16 is\n{medium[2]}")
    print(f"For learning rate = 0.015, iteration = 50, the train accuracy for age 16-18 is\n{high[2]}")

    l_theta, l_beta = low[0], low[1]
    m_theta, m_beta = medium[0], medium[1]
    h_theta, h_beta = high[0], high[1]

    # obtain the test accuracy of male and female algorithm
    l_test_acc = evaluate(l_test, l_theta, l_beta)
    m_test_acc = evaluate(m_test, m_theta, m_beta)
    h_test_acc = evaluate(h_test, h_theta, h_beta)

    print(f"With chosen learning rate = 0.015 and iteration = 50,\n"
          f"The final test accuracy for age 7-13 is {l_test_acc}\n"
          f"The final test accuracy for age 13-16 is {m_test_acc}\n"
          f"The final test accuracy for age 16-18 is {h_test_acc}")

    # run the original irt model
    irt_alg = irt(train_data, val_data, 0.015, 50)

    # plot the train accuracy of male, female against original
    plt.figure(1)
    plt.plot(list(range(50)), boy[2], label="boys train accuracy")
    plt.plot(list(range(50)), girl[2], label="girls train accuracy")
    plt.plot(list(range(50)), irt_alg[2], label="original train accuracy")
    plt.xlabel("iteration")
    plt.ylabel("train accuracy")
    plt.legend()
    plt.figure(1)
    # plot the validation accuracy of male, female against original
    plt.figure(2)
    plt.plot(list(range(50)), boy[3], label="boys validation accuracy")
    plt.plot(list(range(50)), girl[3], label="girls validation accuracy")
    plt.plot(list(range(50)), irt_alg[3], label="original validation accuracy")
    plt.xlabel("iteration")
    plt.ylabel("validation accuracy")
    plt.legend()
    plt.figure(2)
    # plot the train accuracy of different age group against original
    plt.figure(3)
    plt.plot(list(range(50)), low[2], label="age 7-13 train accuracy")
    plt.plot(list(range(50)), medium[2], label="age 13-16 train accuracy")
    plt.plot(list(range(50)), high[2], label="age 16-18 train accuracy")
    plt.plot(list(range(50)), irt_alg[2], label="original train accuracy")
    plt.xlabel("iteration")
    plt.ylabel("train accuracy")
    plt.legend()
    plt.figure(3)
    # plot the validation accuracy of male, female against original
    plt.figure(4)
    plt.plot(list(range(50)), low[3], label="age 7-13 validation accuracy")
    plt.plot(list(range(50)), medium[3], label="age 13-16 validation accuracy")
    plt.plot(list(range(50)), high[3], label="age 16-18 validation accuracy")
    plt.plot(list(range(50)), irt_alg[3], label="original validation accuracy")
    plt.xlabel("iteration")
    plt.ylabel("validation accuracy")
    plt.legend()
    plt.figure(4)


if __name__ == "__main__":
    main()
