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

    val_set_num_feature_matrix = np.delete(np.array(val_set["data"]), vote_feature_index_list, 1)[:,:-1]
    val_set_vote_feature_matrix = np.delete(np.array(val_set["data"]), num_feature_index_list, 1)[:,:-1]
    val_set_label_matrix = np.array(val_set["data"])[:,-1:]

    test_set_num_feature_matrix = np.delete(np.array(test_set["data"]), vote_feature_index_list, 1)[:,:-1]
    test_set_vote_feature_matrix = np.delete(np.array(test_set["data"]), num_feature_index_list, 1)[:,:-1]
    test_set_label_matrix = np.array(test_set["data"])[:,-1:]

    mean = None
    std = None
    normed_training_set_num_feature_matrix = np.array([])
    normed_val_set_num_feature_matrix = np.array([])
    normed_test_set_num_feature_matrix = np.array([])
    # [num,vote]
    feature_status = [0,0]

    if training_set_num_feature_matrix.size != 0 :
        feature_status[0] = 1
        mean = np.mean(training_set_num_feature_matrix,0)
        std = np.std(training_set_num_feature_matrix,0)
        std[std == 0] = 1
        normed_training_set_num_feature_matrix = (training_set_num_feature_matrix.astype(float) - mean)/std.astype(float)
        normed_val_set_num_feature_matrix = (val_set_num_feature_matrix.astype(float) - mean)/std.astype(float)
        normed_test_set_num_feature_matrix = (test_set_num_feature_matrix.astype(float) - mean)/std.astype(float)

    if training_set_vote_feature_matrix.size != 0 :
        feature_status[1] = 1

    # accuracy stored in this column vector, later can be stable sorted
    k_acc_matrix = np.zeros((max_k,1))

    if feature_status == [1,0]:
        for k in range(1,max_k+1):
            for i in range(normed_val_set_num_feature_matrix.shape[0]):

                dict_count = OrderedDict()
                for label in training_set["metadata"]["features"][-1:][0][1]:
                    dict_count[label] = 0

                num_d = np.absolute(normed_training_set_num_feature_matrix - normed_val_set_num_feature_matrix[i])
                num_d = np.sum(num_d, 1)
                knn_index_list = np.argsort(num_d,axis=0,kind="stable")[:k]
                knn_label_list = training_set_label_matrix[knn_index_list]
                for label in knn_label_list:
                    dict_count[label[0]] += 1
                max_count_label = max(dict_count,key=dict_count.get)
                if(max_count_label == val_set_label_matrix[i][0]):
                    k_acc_matrix[k-1] += 1

    if feature_status == [0,1]:
        for k in range(1,max_k+1):
            for i in range(val_set_vote_feature_matrix.shape[0]):

                dict_count = OrderedDict()
                for label in training_set["metadata"]["features"][-1:][0][1]:
                    dict_count[label] = 0

                vote_d = np.sum(val_set_vote_feature_matrix[i] != training_set_vote_feature_matrix, 1)
                knn_index_list = np.argsort(vote_d,axis=0,kind="stable")[:k]
                knn_label_list = training_set_label_matrix[knn_index_list]
                for label in knn_label_list:
                    dict_count[label[0]] += 1

                max_count_label = max(dict_count,key=dict_count.get)
                if(max_count_label == val_set_label_matrix[i][0]):
                    k_acc_matrix[k-1][0] += 1

    if feature_status == [1,1]:
        for k in range(1,max_k+1):
            for i in range(val_set_vote_feature_matrix.shape[0]):

                dict_count = OrderedDict()
                for label in training_set["metadata"]["features"][-1:][0][1]:
                    dict_count[label] = 0
                num_d = np.absolute(normed_training_set_num_feature_matrix - normed_val_set_num_feature_matrix[i])
                num_d = np.sum(num_d, 1)
                vote_d = np.sum(val_set_vote_feature_matrix[i] != training_set_vote_feature_matrix, 1)
                total_d = num_d + vote_d
                knn_index_list = np.argsort(total_d,axis=0,kind="stable")[:k]
                knn_label_list = training_set_label_matrix[knn_index_list]
                for label in knn_label_list:
                    dict_count[label[0]] += 1
                max_count_label = max(dict_count,key=dict_count.get)
                if(max_count_label == val_set_label_matrix[i][0]):
                    k_acc_matrix[k-1][0] += 1

    k_acc_matrix = k_acc_matrix/float(val_set_label_matrix.shape[0])
    for i in range(k_acc_matrix.shape[0]):
        print("" + str((i+1)) +"," + str(k_acc_matrix[i][0]))
    best_k = np.argsort((-k_acc_matrix),axis=0,kind="stable")[0][0] + 1
    best_k_acc = 0

    print(normed_training_set_num_feature_matrix.shape[0])
    normed_training_set_num_feature_matrix = np.vstack((normed_training_set_num_feature_matrix,normed_val_set_num_feature_matrix))
    print(normed_training_set_num_feature_matrix.shape[0])
    training_set_vote_feature_matrix = np.vstack((training_set_vote_feature_matrix,val_set_vote_feature_matrix))
    training_set_label_matrix = np.vstack((training_set_label_matrix,val_set_label_matrix))

    for i in range(test_set_label_matrix.shape[0]):

        dict_count = OrderedDict()
        for label in training_set["metadata"]["features"][-1:][0][1]:
            dict_count[label] = 0

        d = 0
        if feature_status == [1,0]:
            d = np.sum(np.absolute(normed_training_set_num_feature_matrix - normed_test_set_num_feature_matrix[i]), 1)
        if feature_status == [0,1]:
            d = np.sum(test_set_vote_feature_matrix[i] != training_set_vote_feature_matrix, 1)
        if feature_status == [1,1]:
            d = np.sum(np.absolute(normed_training_set_num_feature_matrix - normed_test_set_num_feature_matrix[i]), 1) + np.sum(test_set_vote_feature_matrix[i] != training_set_vote_feature_matrix, 1)
        knn_index_list = np.argsort(d,axis=0,kind="stable")[:best_k]
        knn_label_list = training_set_label_matrix[knn_index_list]
        for label in knn_label_list:
            dict_count[label[0]] += 1
        max_count_label = max(dict_count,key=dict_count.get)
        if(max_count_label == test_set_label_matrix[i][0]):
            best_k_acc += 1

    print(best_k_acc)
    print(test_set_label_matrix.shape[0])
    best_k_acc = float(best_k_acc)/float(test_set_label_matrix.shape[0])
    print(str(best_k))
    print(str(best_k_acc))


if __name__ == "__main__":
    main()
