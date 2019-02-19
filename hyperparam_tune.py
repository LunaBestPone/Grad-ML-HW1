import sys, json
import numpy as np
from collections import OrderedDict

def main():
    if len(sys.argv) != 5:
        print("Usage :   ./hyperparam_tune <max k> <TRAIN SET> <VAL. SET> <TEST SET>")

    if not sys.argv[1].isdigit():
        print("max k must be an INT")

    max_k = int(sys.argv[1])

    training_set_path = sys.argv[2]
    val_set_path = sys.argv[3]
    test_set_path = sys.argv[4]

    training_set_file = open(training_set_path,'r')
    val_set_file = open(val_set_path,'r')
    test_set_file = open(test_set_path,'r')

    training_set = json.load(training_set_file)
    val_set = json.load(val_set_file)
    test_set = json.load(test_set_file)

    for idx, feature in enumerate(training_set["metadata"]["features"][:-1]):
        if feature[1] == "numeric":
            num_feature_index_list.append(idx)
        else:
            vote_feature_index_list.append(idx)

    training_set_num_feature_matrix = np.delete(np.array(training_set["data"]), vote_feature_index_list, 1)[:,:-1]
    training_set_vote_feature_matrix = np.delete(np.array(training_set["data"]), num_feature_index_list, 1)[:,:-1]
    training_set_label_matrix = np.array(training_set["data"])[:,-1:]

    val_set_num_feature_matrix = np.delete(np.array(val_set["data"]), vote_feature_index_list, 1)[:,:-1]
    val_set_vote_feature_matrix = np.delete(np.array(val_set["data"]), num_feature_index_list, 1)[:,:-1]
    val_set_label_matrix = np.array(val_set["data"])[:,-1:]

    test_set_num_feature_matrix = np.delete(np.array(test_set["data"]), vote_feature_index_list, 1)[:,:-1]
    test_set_vote_feature_matrix = np.delete(np.array(test_set["data"]), num_feature_index_list, 1)[:,:-1]
    test_set_label_matrix = np.array(test_set["data"])[:,-1:]

if __name__ = "__main__":
    main()
