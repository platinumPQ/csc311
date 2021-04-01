from sklearn.impute import KNNImputer
from final_project.starter_code.utils import *
import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("For k = {}, validation accuracy imputed by user is: {}".format(k,acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)

    # Using KNN based on question similarity. Using matrix transpose,
    # we measure the nan-euclidean distance of column vectors
    inter = nbrs.fit_transform(matrix.T)
    mat = inter.T
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("For k = {}, validation accuracy imputed by item is: {}".format(k, acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    acc_by_user = []
    acc_by_item = []
    k_list = [1, 6, 11, 16, 21, 26]
    for k in k_list:
        acc_by_user.append(knn_impute_by_user(sparse_matrix, val_data, k))
        acc_by_item.append(knn_impute_by_item(sparse_matrix, val_data, k))
    print(f"Knn imputed by user have accuracy\n{acc_by_user}")
    print(f"Knn imputed by question have accuracy\n{acc_by_item}")

    # plot the validation accuracy as a function of k
    plt.figure(1)
    plt.plot(k_list, acc_by_user)
    plt.xlabel("k - knn hyperparameter")
    plt.ylabel("accuracy on validation")
    plt.title("Validation accuracy against K imputed by user")
    plt.show()
    plt.figure(1)

    plt.figure(2)
    plt.plot(k_list, acc_by_item)
    plt.xlabel("k - knn hyperparameter")
    plt.ylabel("accuracy on validation")
    plt.title("Validation accuracy against K imputed by question")
    plt.show()
    plt.figure(2)
    # report the best accuracy k
    k_user = k_list[acc_by_user.index(max(acc_by_user))]
    k_item = k_list[acc_by_item.index(max(acc_by_item))]
    print(f"For knn imputed by users, k* = {k_user} has the highest performance on validation data."
          f"\nThe final test accuracy is {knn_impute_by_user(sparse_matrix, test_data, k_user)}.")
    print(f"For knn imputed by questions, k* = {k_item} has the highest performance on validation data."
          f"\nThe final test accuracy is {knn_impute_by_item(sparse_matrix, test_data, k_item)}.")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
