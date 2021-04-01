from utils import *


def _load_csv(path):
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {
        "user_id": [],
        "gender": [],
        "data_of_birth": []
    }
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data["user_id"].append(int(row[0]))
                data["gender"].append(int(row[1]))
                data["data_of_birth"].append(row[2])
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data


def load_student_meta_csv(root_dir="../data"):
    """ Load the student_meta data as a dictionary.

    :param root_dir: str
    # :return: A dictionary {user_id: list, question_id: list, is_correct: list}
    #     WHERE
    #     user_id: a list of user id.
    #     question_id: a list of question id.
    #     is_correct: a list of binary value indicating the correctness of
    #     (user_id, question_id) pair.
    """
    path = os.path.join(root_dir, "student_meta.csv")
    return _load_csv(path)


def separate_gender():
    dic = load_student_meta_csv()
    id_list = dic["user_id"]
    gender_list = dic["gender"]

    male_list = []
    female_list = []
    for i in range(len(gender_list)):
        if gender_list[i] == 1:
            female_list.append(id_list[i])
        elif gender_list[i] == 2:
            male_list.append(id_list[i])

    return male_list, female_list


def get_dic(male_list, female_list, original_dic):
    m_dic= {"user_id": [], "question_id": [], "is_correct": []}
    f_dic = {"user_id": [], "question_id": [], "is_correct": []}
    original_id = original_dic["user_id"]
    original_question = original_dic["question_id"]
    original_correct = original_dic["is_correct"]

    for i in range(len(original_id)):
        if original_id[i] in male_list:
            m_dic["user_id"].append(original_id[i])
            m_dic["question_id"].append(original_question[i])
            m_dic["is_correct"].append(original_correct[i])
        elif original_id[i] in female_list:
            f_dic["user_id"].append(original_id[i])
            f_dic["question_id"].append(original_question[i])
            f_dic["is_correct"].append(original_correct[i])

    return m_dic, f_dic


if __name__ == '__main__':
    m, f = separate_gender()
    train_data_dic = load_train_csv("../data")
    valid_data_dic = load_valid_csv("../data")
    test_data_dic = load_public_test_csv("../data")
    md, fd = get_dic(m, f, train_data_dic)
    print(md)

