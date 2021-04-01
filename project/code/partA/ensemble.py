# TODO: complete this file.
from utils import *
from neural_network import *
from torch.autograd import Variable

import torch.utils.data

import numpy as np
import torch
import matplotlib.pyplot as plt


def select_data():
    """
    Return a group of random selected training data, and return its model trained with hyperparameters selected
    from previous part.
    """
    # zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    num_students = train_matrix.shape[0]
    # np.random.seed(1)
    sample = np.random.randint(0, num_students, num_students)

    # set of hyperparameters with highest accuracy from part a - 3
    k = 50
    lr = 0.05
    num_epoch = 10
    lamb = 0.1

    num_questions = train_matrix.shape[1]

    # k is the number of latent dimensions with highest validation accuracy
    evaluated_model = AutoEncoder(num_questions, k)

    sampled_train_matrix = np.zeros(train_matrix.shape)
    sampled_zero_train_matrix = np.zeros(zero_train_matrix.shape)

    for i in range(num_students):
        selected_index = sample[i]
        sampled_train_matrix[i] = train_matrix[selected_index]
        sampled_zero_train_matrix[i] = zero_train_matrix[selected_index]

    # train_tensor = torch.from_numpy(sampled_train_matrix)
    # zero_train_tensor = torch.from_numpy(sampled_zero_train_matrix)
    train_tensor = torch.FloatTensor(sampled_train_matrix)
    zero_train_tensor = torch.FloatTensor(sampled_zero_train_matrix)
    # train
    train(evaluated_model, lr, lamb, train_tensor, zero_train_tensor, valid_data, num_epoch)

    # test_accuracy = evaluate(model, zero_train_tensor, test_data)
    # print(test_accuracy)
    return evaluated_model, zero_train_tensor


def evaluate_bagging(evaluated_model, train_data_1, train_data_2, train_data_3, testing_data):
    """ Evaluate the valid_data on the current model.

    :param evaluated_model: Module
    :param train_data_1: 2D FloatTensor
    :param train_data_2: 2D FloatTensor
    :param train_data_3: 2D FloatTensor
    :param testing_data: Either validation or test data. A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    evaluated_model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(testing_data["user_id"]):
        inputs_1 = Variable(train_data_1[u]).unsqueeze(0)
        output_1 = evaluated_model(inputs_1)
        inputs_2 = Variable(train_data_2[u]).unsqueeze(0)
        output_2 = evaluated_model(inputs_2)
        inputs_3 = Variable(train_data_3[u]).unsqueeze(0)
        output_3 = evaluated_model(inputs_3)

        guess = 1 / 3 * (output_1[0][testing_data["question_id"][i]].item() +
                         output_2[0][testing_data["question_id"][i]].item() +
                         output_3[0][testing_data["question_id"][i]].item()) >= 0.5
        if guess == testing_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def bagging():
    pass


if __name__ == "__main__":
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    model, zero_train_tensor_1 = select_data()
    zero_train_tensor_2 = select_data()[1]
    zero_train_tensor_3 = select_data()[1]
    vali_accuracy = evaluate_bagging(model, zero_train_tensor_1, zero_train_tensor_2, zero_train_tensor_3, valid_data)
    test_accuracy = evaluate_bagging(model, zero_train_tensor_1, zero_train_tensor_2, zero_train_tensor_3, test_data)
    print(vali_accuracy)
    print(test_accuracy)
