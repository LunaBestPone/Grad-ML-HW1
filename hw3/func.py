import numpy as np
import sys,json

# load data from path using json load
def load_data(training_set_path, test_set_path):
    training_set_file = open(training_set_path,'r')
    test_set_file = open(test_set_path,'r')

    training_set = json.load(training_set_file)
    test_set = json.load(test_set_file)

    training_set_file.close()
    test_set_file.close()

    return training_set, test_set

# find the index of num and category features
def classify_features(training_set, test_set):
    num_feature_index_list = []
    cate_feature_index_list = []

    for idx, feature in enumerate(training_set["metadata"]["features"][:-1]):
        if feature[1] == "numeric":
            num_feature_index_list.append(idx)
        else:
            cate_feature_index_list.append(idx)

    return num_feature_index_list, cate_feature_index_list

# get training or test set matrices from given set
def get_matrices(set, num_feature_index_list, cate_feature_index_list):
    set_num_feature_matrix = np.delete(np.array(set["data"]), cate_feature_index_list, 1)[:,:-1]
    set_num_feature_matrix = set_num_feature_matrix.astype(np.float)
    set_cate_feature_matrix = np.delete(np.array(set["data"]), num_feature_index_list, 1)[:,:-1]
    set_label_matrix = np.array(set["data"])[:,-1:]

    return set_num_feature_matrix, set_cate_feature_matrix, set_label_matrix

# standardization of num features
def standardize(training_set_num_feature_matrix, test_set_num_feature_matrix):
    mean = np.mean(training_set_num_feature_matrix,0)
    std = np.std(training_set_num_feature_matrix,0)
    std[std == 0] = 1
    normed_training_set_num_feature_matrix = (training_set_num_feature_matrix.astype(float) - mean)/std.astype(float)
    normed_test_set_num_feature_matrix = (test_set_num_feature_matrix.astype(float) - mean)/std.astype(float)

    return normed_training_set_num_feature_matrix, normed_test_set_num_feature_matrix

# sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))
