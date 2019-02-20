import sys, json
import numpy as np
from collections import OrderedDict
from operator import itemgetter

def main():
    if len(sys.argv) != 4:
        print("Usage :   ./learning_curve <INT k> <TRAINING SET> <TEST SET>")
        sys.exit(0)

    if not sys.argv[1].isdigit():
        print("k must be an INT")
        sys.exit(0)

    k = int(sys.argv[1])

    training_set_path = sys.argv[2]
    test_set_path = sys.argv[3]

    training_set_file = open(training_set_path,'r')
    test_set_file = open(test_set_path,'r')

    training_set = json.load(training_set_file)
    test_set = json.load(test_set_file)

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

    if training_set_vote_feature_matrix.size != 0 :
        feature_status[1] = 1


    full_length = training_set_label_matrix.shape[0]
    for f in range(1,11):
        cur_length = int(full_length*(f/10))
        cur_training_set_num_feature_matrix = np.delete(training_set_num_feature_matrix,range(cur_length,full_length-1),0)
        if feature_status[0] == 1:
            mean = np.mean(cur_training_set_num_feature_matrix,0)
            std = np.std(cur_training_set_num_feature_matrix,0)
            std[std==0] = 1
            normed_training_set_num_feature_matrix = (cur_training_set_num_feature_matrix.astype(float) - mean)/std.astype(float)
            normed_test_set_num_feature_matrix = (test_set_num_feature_matrix.astype(float) - mean)/std.astype(float)
        cur_training_set_vote_feature_matrix = np.delete(training_set_vote_feature_matrix,range(cur_length,full_length-1),0)
        cur_training_set_label_matrix = np.delete(training_set_label_matrix,range(cur_length,full_length-1),0)
        acc = 0
        for i in range(test_set_label_matrix.shape[0]):
            dict_count = OrderedDict()
            for label in training_set["metadata"]["features"][-1:][0][1]:
                dict_count[label] = 0
            d = 0
            if feature_status == [1,0]:
                d = np.sum(np.absolute(normed_training_set_num_feature_matrix - normed_test_set_num_feature_matrix[i]), 1)
            if feature_status == [0,1]:
                d = np.sum(test_set_vote_feature_matrix[i] != cur_training_set_vote_feature_matrix, 1)
            if feature_status == [1,1]:
                d = np.sum(np.absolute(normed_training_set_num_feature_matrix - normed_test_set_num_feature_matrix[i]), 1) + np.sum(test_set_vote_feature_matrix[i] != cur_training_set_vote_feature_matrix, 1)
            knn_index_list = np.argsort(d,axis=0,kind="stable")[:k]
            knn_label_list = cur_training_set_label_matrix[knn_index_list]
            for label in knn_label_list:
                dict_count[label[0]] += 1
            max_count_label = sorted(dict_count.items(),key=itemgetter(1),reverse=True)[0][0]
            if(max_count_label == test_set_label_matrix[i][0]):
                acc += 1
        acc = float(acc)/float(test_set_label_matrix.shape[0])
        print(str(cur_length) +","+str(acc))

if __name__ ==  "__main__":
    main()
