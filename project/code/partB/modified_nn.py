from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch
import matplotlib.pyplot as plt


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        question_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        question_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.s = nn.Linear(k, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1|| + ||W^2||.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2)
        s_w_norm = torch.norm(self.s.weight, 2)
        h_w_norm = torch.norm(self.h.weight, 2)
        return g_w_norm + s_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        activate_g = self.g(inputs)
        output1 = F.sigmoid(activate_g)
        # output = torch.sigmoid(activate_g)
        activate_s = self.s(output1)
        output2 = F.sigmoid(activate_s)
        activate_f = self.h(output2)
        out = F.sigmoid(activate_f)
        # out = torch.sigmoid(activate_f)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function. 

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    # load the dic for train data
    train_data_dic = load_train_csv("../data")
    valid_list = []
    train_list = []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            # loss = torch.sum((output - target) ** 2.)
            # loss.backward()
            # modified loss for part (e), based on given L2 formula
            L2 = (lamb/2)*(model.get_weight_norm())
            l2_loss = torch.sum((output - target) ** 2) + L2
            l2_loss.backward()

            # train_loss += loss.item()
            train_loss += l2_loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)

        # loss
        valid_loss = compute_loss(model, zero_train_data, valid_data)
        train_loss_com = compute_loss(model, zero_train_data, train_data_dic)
        valid_list.append(valid_loss)
        train_list.append(train_loss_com)

        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
    return valid_list, train_list
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def compute_loss(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model, for loss

        :param model: Module
        :param train_data: 2D FloatTensor
        :param valid_data: A dictionary {user_id: list,
        question_id: list, is_correct: list}
        :return: float
        """
    # similar as evaluate above
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    total_num = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)
        # get target 'label'
        label = valid_data['is_correct'][i]
        # get predict
        question_id = valid_data['question_id'][i]
        predict = output[0][question_id].item()
        # for counting avg
        total_num += 1
        # cost for this loop
        total += (predict - label) ** 2

    # print(total)
    # print(total_num)
    return total / total_num


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    # shape: ([542, 1774]), ([542, 1774])
    # print(valid_data.keys())    # dict_keys(['user_id', 'question_id', 'is_correct'])
    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    num_question = train_matrix.shape[1]
    # k list: [10, 50, 100, 200, 500]
    k = 50
    model = AutoEncoder(num_question, k)

    # Set optimization hyperparameters.
    # tried learning rate [0.01, 0.05, 0.1, 0.5]
    lr = 0.05  # learning rate
    num_epoch = 25  # iteration?
    # lambda list = [0.001, 0.01, 0.1, 1]
    lamb = 0.1

    valid_list, train_list = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)

    test_accuracy = evaluate(model, zero_train_matrix, test_data)
    print(test_accuracy)

    # plot for part d
    plt.title('Training Loss vs. Validation Loss over Epochs')
    plt.plot(train_list, color='blue', label='Training Loss')
    plt.plot(valid_list, color='orange', label='Validation Loss')
    plt.legend(loc='best')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epochs')
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
