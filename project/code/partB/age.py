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
                data["data_of_birth"].append(row[2][:4])
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


def separate_age():
    """
    Separate age groups (given from 7-18 years old) to 7-12, 13-15, 16-18.
    Basically based on school levels.
    ie. primary school, middle school, high school.
    """
    dic = load_student_meta_csv()
    id_list = dic["user_id"]
    dob_list = dic["data_of_birth"]
    prim_list = []
    mid_list = []
    high_list = []

    for i in range(len(id_list)):
        if dob_list[i][:4] != "":
            year = int(dob_list[i][:4])
            age = 2020 - year

            if 7 <= age < 13:
                prim_list.append(id_list[i])
            elif 13 <= age < 16:
                mid_list.append(id_list[i])
            elif 16 <= age <= 18:
                high_list.append(id_list[i])

    return prim_list, mid_list, high_list


def get_dict(prim_list, mid_list, high_list, original_dic):
    p_dic = {"user_id": [], "question_id": [], "is_correct": []}
    m_dic = {"user_id": [], "question_id": [], "is_correct": []}
    h_dic = {"user_id": [], "question_id": [], "is_correct": []}
    original_id = original_dic["user_id"]
    original_question = original_dic["question_id"]
    original_correct = original_dic["is_correct"]

    for i in range(len(original_id)):
        if original_id[i] in prim_list:
            p_dic["user_id"].append(original_id[i])
            p_dic["question_id"].append(original_question[i])
            p_dic["is_correct"].append(original_correct[i])
        elif original_id[i] in mid_list:
            m_dic["user_id"].append(original_id[i])
            m_dic["question_id"].append(original_question[i])
            m_dic["is_correct"].append(original_correct[i])
        elif original_id[i] in high_list:
            h_dic["user_id"].append(original_id[i])
            h_dic["question_id"].append(original_question[i])
            h_dic["is_correct"].append(original_correct[i])

    return p_dic, m_dic, h_dic


if __name__ == '__main__':
    # dic = load_student_meta_csv()
    # print(dic["data_of_birth"])
    prim_list, mid_list, high_list = separate_age()
    # print(prim_list)
    train_data_dic = load_train_csv("../data")
    valid_data_dic = load_valid_csv("../data")
    test_data_dic = load_public_test_csv("../data")
    p_dic, m_dic, h_dic = get_dict(prim_list, mid_list, high_list, test_data_dic)
    print(p_dic)
