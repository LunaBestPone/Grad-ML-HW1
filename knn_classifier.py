import sys, json
import numpy as np
from collections import OrderedDict

def main():
    if len(sys.argv) != 4:
        print("Usage :  ./knn_classifier <INT k> <TRAINING SET> <TEST SET>")

    if not sys.argv[1].isdigit():
        print("k must be an INT")

    k = int(sys.argv[1])
    training_set_path = sys.argv[2]
    test_set_path = sys.argv[3]

    training_set_file = open(training_set_path,'r')
    test_set_file = open(test_set_path,'r')

    training_set = json.load(training_set_file)
    test_set = json.load(test_set_file)

    training_set_file.close()
    test_set_file.close()

    num_feature_index_list = []
    vote_feature_index_list = []

    for idx, feature in enumerate(training_set["metadata"]["features"][:-1]):
        if feature[1] == "numeric":
            num_feature_index_list.append(idx)
        else:
            vote_feature_index_list.append(idx)

    training_set_num_feature_matrix = np.delete(np.array(training_set["data"]), vote_feature_index_list, 1)[:,:-1]
    training_set_vote_feature_matrix = np.delete(np.array(training_set["data"]), num_feature_index_list, 1)[:,:-1]
    training_set_label_matrix = np.array(training_set["data"])[:,-1:]

    test_set_num_feature_matrix = np.delete(np.array(test_set["data"]), vote_feature_index_list, 1)[:,:-1]
    test_set_vote_feature_matrix = np.delete(np.array(test_set["data"]), num_feature_index_list, 1)[:,:-1]
    test_set_label_matrix = np.array(test_set["data"])[:,-1:]

    mean = None
    std = None
    normed_training_set_num_feature_matrix = np.array([])
    normed_test_set_num_feature_matrix = np.array([])
    # [num,vote]
    feature_status = [0,0]

    if training_set_num_feature_matrix.size != 0 :
        feature_status[0] = 1
        mean = np.mean(training_set_num_feature_matrix,0)
        std = np.std(training_set_num_feature_matrix,0)
        std[std == 0] = 1
        normed_training_set_num_feature_matrix = (training_set_num_feature_matrix.astype(float) - mean)/std.astype(float)
        normed_test_set_num_feature_matrix = (test_set_num_feature_matrix.astype(float) - mean)/std.astype(float)

    if training_set_vote_feature_matrix.size != 0 :
        feature_status[1] = 1

    if feature_status == [1,0]:
        for instance in normed_test_set_num_feature_matrix:

            dict_count = OrderedDict()
            for label in training_set["metadata"]["features"][-1:][0][1]:
                dict_count[label] = 0

            num_d = np.absolute(normed_training_set_num_feature_matrix - instance)
            num_d = np.sum(num_d, 1)
            knn_index_list = np.argsort(num_d,axis=0,kind="stable")[:k]
            knn_label_list = training_set_label_matrix[knn_index_list]
            for label in knn_label_list:
                dict_count[label[0]] += 1
            outstr = ""
            for label,count in dict_count.items():
                outstr += str(count) + ","
            max_count_label = max(dict_count,key=dict_count.get)
            outstr += str(max_count_label)
            print(outstr)

    if feature_status == [0,1]:
        for instance in test_set_vote_feature_matrix:

            dict_count = OrderedDict()
            for label in training_set["metadata"]["features"][-1:][0][1]:
                dict_count[label] = 0

            vote_d = np.sum(instance != training_set_vote_feature_matrix, 1)
            knn_index_list = np.argsort(vote_d,axis=0,kind="stable")[:k]
            knn_label_list = training_set_label_matrix[knn_index_list]
            for label in knn_label_list:
                dict_count[label[0]] += 1
            outstr = ""
            for label,count in dict_count.items():
                outstr += str(count) + ","
            max_count_label = max(dict_count,key=dict_count.get)
            outstr += str(max_count_label)
            print(outstr)

    if feature_status == [1,1]:
        for i in range(normed_test_set_num_feature_matrix.shape[0]):

            dict_count = OrderedDict()
            for label in training_set["metadata"]["features"][-1:][0][1]:
                dict_count[label] = 0

            num_d = np.absolute(normed_training_set_num_feature_matrix - normed_test_set_num_feature_matrix[i])
            num_d = np.sum(num_d, 1)
            vote_d = np.sum(test_set_vote_feature_matrix[i] != training_set_vote_feature_matrix, 1)
            total_d = num_d + vote_d
            knn_index_list = np.argsort(total_d,axis=0,kind="stable")[:k]
            knn_label_list = training_set_label_matrix[knn_index_list]
            for label in knn_label_list:
                dict_count[label[0]] += 1
            outstr = ""
            for label,count in dict_count.items():
                outstr += str(count) + ","
            max_count_label = max(dict_count,key=dict_count.get)
            outstr += str(max_count_label)
            print(outstr)


if __name__ == "__main__":
    main()
