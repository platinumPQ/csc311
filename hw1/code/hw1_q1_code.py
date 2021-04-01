import sklearn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def load_data(file_fake, file_real):
    """
    load the input data, preprocess using CountVectorizer and return the splitted 70% training, 15% validation and
    15% test examples of the resulting dataset
    """
    # load the data
    fake_dt = [line.strip() for line in open(file_fake, 'r')]
    real_dt = [line.strip() for line in open(file_real, 'r')]
    dataset = fake_dt + real_dt
    label = [1] * len(fake_dt) + [0] * len(real_dt)
    with_label = [(item, 1)for item in fake_dt] + [(item, 0)for item in real_dt]

    # vectorize the text file into a sparse matrix
    vec = CountVectorizer()
    headlines = vec.fit_transform(dataset)
    hl_names = vec.get_feature_names()

    # randomize dataset order and split into training, test, and validation sets
    hl_train, hl_temp, label_train, label_temp = train_test_split(headlines, label, test_size=0.3, random_state=30)
    hl_test, hl_val, label_test, label_val = train_test_split(hl_temp, label_temp, test_size=0.5, random_state=30)

    return hl_train, hl_test, hl_val, label_train, label_test, label_val


def select_knn_model(trainset, valset, trainlabel, vallabel, metric='minkowski'):
    """
    Return the k value with the highest validation accuracy and the corresponding train and validation error.
    """
    train_accuracy, train_error, val_accuracy, val_error = [], [], [], []
    best_acc = 0

    for i in range(1, 21):
        neigh = KNeighborsClassifier(metric=metric, n_neighbors=i)
        neigh.fit(trainset, trainlabel)

        # compute the mean training errors and accuracy
        train_acc = neigh.score(trainset, trainlabel)
        train_accuracy.append(train_acc)
        train_error.append(1-train_acc)

        # compute the mean validation errors and accuracy
        val_acc = neigh.score(valset, vallabel)
        val_accuracy.append(val_acc)
        val_error.append(1 - val_acc)

        # find the model with the highest validation accuracy to tune the hyperparameter k
        if best_acc <= val_acc:
            best_acc, best_k = val_acc, i

    # plot the training and validation accuracy for each k
    plt.plot(range(1, 21), train_accuracy, label='training with metric=' + metric)
    plt.plot(range(1, 21), val_accuracy, label='validation with metric=' + metric)
    plt.xlabel('k - number of nearest neighbour')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    return best_k, train_error, val_error


if __name__ == "__main__":
    interactive(False)
    file_fake = 'D:\AAAAAA YUPEIQING\\4\csc311\data\clean_fake.txt'
    file_real = 'D:\AAAAAA YUPEIQING\\4\csc311\data\clean_real.txt'

    hl_train, hl_test, hl_val, label_train, label_test, label_val = load_data(file_fake, file_real)

    # find the tuned hyperparameter
    k_tuned, train_err_ori, val_err_ori = select_knn_model(hl_train, hl_val, label_train, label_val)
    knn_model = KNeighborsClassifier(n_neighbors=k_tuned)
    knn_model.fit(hl_test, label_test)
    knn_model.score(hl_test, label_test)

    # using metric=cosine to compute the distance between data points
    k_new, train_err_new, val_err_new = select_knn_model(hl_train, hl_val, label_train, label_val, metric='cosine')
    new_model = KNeighborsClassifier(n_neighbors=k_tuned)
    new_model.fit(hl_test, label_test)
    new_model.score(hl_test, label_test)

    print(train_err_ori, train_err_new,val_err_ori, val_err_new)
