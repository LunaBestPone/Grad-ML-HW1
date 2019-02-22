import sys, json
import numpy as np
from collections import OrderedDict
from operator import itemgetter

def main():
    if len(sys.argv) != 4:
        print("Usage :    ./roc_curve <INT k> <TRAINING SET> <TEST SET>")
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

    positive_label = training_set["metadata"]["features"][-1][1][0]

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

    confidence = np.zeros((test_set_label_matrix.shape[0],1),dtype="float_")

    for i in range(test_set_label_matrix.shape[0]):

        if feature_status == [1,0]:
            d = np.sum(np.absolute(normed_training_set_num_feature_matrix - normed_test_set_num_feature_matrix[i]), 1)
        if feature_status == [0,1]:
            d = np.sum(test_set_vote_feature_matrix[i] != training_set_vote_feature_matrix, 1)
        if feature_status == [1,1]:
            d = np.sum(np.absolute(normed_training_set_num_feature_matrix - normed_test_set_num_feature_matrix[i]), 1) + np.sum(test_set_vote_feature_matrix[i] != training_set_vote_feature_matrix, 1)
        knn_index_list = np.argsort(d,axis=0,kind="stable")[:k]
        knn_d = d[knn_index_list]
        knn_d2 = np.power(knn_d,2,dtype="float_")
        wn = np.reciprocal((knn_d2+0.00001),dtype="float_")
        knn_label_list = training_set_label_matrix[knn_index_list]
        knn_class_list = np.zeros((knn_label_list.shape[0],1),dtype="float_")
        for j in range(knn_label_list.shape[0]):
            if knn_label_list[j] == positive_label:
                knn_class_list[j] = 1
            else:
                knn_class_list[j] = 0
        upper = np.dot(wn,knn_class_list)
        lower = np.sum(wn,dtype="float_")
        confidence[i][0] = upper[0]/lower

    confidence_sorted_arg = np.argsort(-confidence,axis=0,kind="stable")
    sorted_confidence = confidence[confidence_sorted_arg]
    num_pos = np.sum(test_set_label_matrix == positive_label,axis=0)
    num_neg = np.sum(test_set_label_matrix != positive_label,axis=0)
    tp = 0
    fp = 0
    last_tp = 0
    for i in range(test_set_label_matrix.shape[0]):
        if i > 1 and sorted_confidence[i] != sorted_confidence[i-1] and test_set_label_matrix[confidence_sorted_arg[i]] != positive_label and tp>last_tp:
            fpr = fp/float(num_neg)
            tpr = tp/float(num_pos)
            print(str(fpr)+","+str(tpr))
            last_tp = tp
        if test_set_label_matrix[confidence_sorted_arg[i]] == positive_label:
            tp += 1
        else:
            fp += 1
    fpr = fp/float(num_neg)
    tpr = tp/float(num_pos)
    print(str(fpr)+","+str(tpr))

if __name__ == "__main__":
    main()
