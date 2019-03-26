import numpy as np
import sys,json
import func
import matplotlib.pyplot as plt

def main():
    # simple arguments check
    if len(sys.argv) != 6:
        print("wrong number of arguments")
        sys.exit(0)

    # passing args
    learning_rate = sys.argv[1]
    num_hidden_units = sys.argv[2]
    max_epoch = sys.argv[3]
    training_set_path = sys.argv[4]
    test_set_path = sys.argv[5]

    # load data
    training_set, test_set = func.load_data(training_set_path, test_set_path)
    # find index of num and category features
    num_feature_index_list, cate_feature_index_list = func.classify_features(training_set, test_set)
    # getting traning set matrices
    training_set_num_feature_matrix, training_set_cate_feature_matrix, training_set_label_matrix = func.get_matrices(training_set, num_feature_index_list, cate_feature_index_list)
    test_set_num_feature_matrix, test_set_cate_feature_matrix, test_set_label_matrix = func.get_matrices(test_set, num_feature_index_list, cate_feature_index_list)
    # get the total number of training and test instances
    total_num_training = training_set_num_feature_matrix.shape[0]
    total_num_test = test_set_num_feature_matrix.shape[0]
    # fill up the feature status list
    # [num, cate]
    feature_status = [0,0]
    if training_set_num_feature_matrix.size != 0 :
        feature_status[0] = 1
    if training_set_label_matrix.size != 0 :
        feature_status[1] = 1
    # standardize num features
    normed_training_set_num_feature_matrix = np.zeros((total_num_training,0))
    normed_test_set_num_feature_matrix = np.zeros((total_num_test,0))
    if feature_status[0] == 1:
        normed_training_set_num_feature_matrix, normed_test_set_num_feature_matrix = func.standardize(training_set_num_feature_matrix, test_set_num_feature_matrix)

    # combine the feature matrix, reorder
    combined_index_list = num_feature_index_list + cate_feature_index_list
    sorted_index_list = np.argsort(combined_index_list)
    combined_training_set_feature_matrix = np.hstack((normed_training_set_num_feature_matrix,training_set_cate_feature_matrix))
    combined_test_set_feature_matrix = np.hstack((normed_test_set_num_feature_matrix,test_set_cate_feature_matrix))
    ordered_training_set_feature_matrix = combined_training_set_feature_matrix[:,sorted_index_list]
    ordered_test_set_feature_matrix = combined_test_set_feature_matrix[:,sorted_index_list]

    # one-hot
    num_to_skip = 0
    for idx,original_idx in enumerate(cate_feature_index_list):
        variant_list = training_set["metadata"]["features"][:-1][original_idx][1]

        cur_training_col = training_set_cate_feature_matrix[:,idx]
        cur_test_col = test_set_cate_feature_matrix[:,idx]

        for jdx, variant in enumerate(variant_list):
            cur_training_col[cur_training_col == variant] = jdx
            cur_test_col[cur_test_col == variant] = jdx

        cur_training_col = cur_training_col.astype(int)
        cur_test_col = cur_test_col.astype(int)

        expanded_training_cols = np.zeros((total_num_training,len(variant_list)))
        expanded_training_cols[np.arange(total_num_training),cur_training_col.flatten()] = 1
        expanded_test_cols = np.zeros((total_num_test,len(variant_list)))
        expanded_test_cols[np.arange(total_num_test),cur_test_col.flatten()] = 1

        ordered_training_set_feature_matrix = np.delete(ordered_training_set_feature_matrix,original_idx + num_to_skip,axis=1)
        ordered_training_set_feature_matrix = np.insert(ordered_training_set_feature_matrix,[original_idx + num_to_skip],expanded_training_cols,axis=1)
        ordered_test_set_feature_matrix = np.delete(ordered_test_set_feature_matrix,original_idx + num_to_skip,axis=1)
        ordered_test_set_feature_matrix = np.insert(ordered_test_set_feature_matrix,[original_idx + num_to_skip],expanded_test_cols,axis=1)
        num_to_skip += (len(variant_list) - 1)

    # append bias entry
    ordered_training_set_feature_matrix = np.insert(ordered_training_set_feature_matrix,0,1,axis=1).astype(float)
    ordered_test_set_feature_matrix = np.insert(ordered_test_set_feature_matrix,0,1,axis=1).astype(float)
    # nn SGD
    class_list = training_set["metadata"]["features"][-1][1]
    F1_training = []
    F1_test = []
    for num_epoches in range(1,int(max_epoch)+1):
        # initialize weight
        w_i_h = np.random.uniform(low=-0.01, high=0.01, size=(int(num_hidden_units), ordered_training_set_feature_matrix.shape[1]))
        w_h_o = np.random.uniform(low=-0.01, high=0.01, size=(1, int(num_hidden_units) + 1))
        for epoch in range(num_epoches):
            for idx in range(total_num_training):
                # index indicate hidden unit, 1d
                net_i_h = np.dot(ordered_training_set_feature_matrix[idx,:],np.transpose(w_i_h))
                h = func.sigmoid(net_i_h)
                # adding bias entry
                h_o = np.insert(h,0,1).astype(float)
                net_h_o = np.dot(w_h_o, h_o)
                o = func.sigmoid(net_h_o)
                y = training_set_label_matrix[idx,0]
                if class_list.index(y) == 0:
                    y = 0
                else:
                    y = 1
                d_o = y - o
                d_h = h_o*(1 - h_o)*d_o*w_h_o
                update_h_o = float(learning_rate)*d_o*h_o
                update_i_h = float(learning_rate)*d_h[:,1]*ordered_training_set_feature_matrix[idx,:]
                for curcol in range(2,d_h.shape[1]):
                    temp = float(learning_rate)*d_h[:,curcol]*ordered_training_set_feature_matrix[idx,:]
                    update_i_h = np.vstack((update_i_h,temp))
                w_i_h += update_i_h
                w_h_o += update_h_o
        # prediction on test set
        num_corr = 0
        num_incorr = 0
        # true positive
        tp = 0
        # predicted positive
        pp = 0
        for idx in range(total_num_test):
            # index indicate hidden unit, 1d
            net_i_h = np.dot(ordered_test_set_feature_matrix[idx,:],np.transpose(w_i_h))
            h = func.sigmoid(net_i_h)
            # adding bias entry
            h_o = np.insert(h,0,1).astype(float)
            net_h_o = np.dot(w_h_o, h_o)
            o = func.sigmoid(net_h_o)
            y = test_set_label_matrix[idx,0]
            if class_list.index(y) == 0:
                y = 0
            else:
                y = 1

            pred = 0
            if o > 0.5:
                pred = 1
                pp += 1
            else:
                pred = 0

            if pred == y:
                num_corr +=1
                if pred == 1:
                    tp += 1
            else:
                num_incorr +=1
        actual_pos = np.sum(test_set_label_matrix == class_list[1])
        recall = tp/actual_pos
        precision = tp/pp
        F1 = 2*precision*recall/(precision + recall)
        F1_test.append(F1)

        # prediction on training set
        num_corr = 0
        num_incorr = 0
        # true positive
        tp = 0
        # predicted positive
        pp = 0
        for idx in range(total_num_training):
            # index indicate hidden unit, 1d
            net_i_h = np.dot(ordered_training_set_feature_matrix[idx,:],np.transpose(w_i_h))
            h = func.sigmoid(net_i_h)
            # adding bias entry
            h_o = np.insert(h,0,1).astype(float)
            net_h_o = np.dot(w_h_o, h_o)
            o = func.sigmoid(net_h_o)
            y = training_set_label_matrix[idx,0]
            if class_list.index(y) == 0:
                y = 0
            else:
                y = 1

            pred = 0
            if o > 0.5:
                pred = 1
                pp += 1
            else:
                pred = 0

            if pred == y:
                num_corr +=1
                if pred == 1:
                    tp += 1
            else:
                num_incorr +=1
        actual_pos = np.sum(training_set_label_matrix == class_list[1])
        recall = tp/actual_pos
        precision = tp/pp
        F1 = 2*precision*recall/(precision + recall)
        F1_training.append(F1)

    plt.plot(range(1,int(max_epoch)+1),F1_training,label="on training set")
    plt.plot(range(1,int(max_epoch)+1),F1_test,label="on test set")
    plt.title("F1 vs #epoches on heart dataset, learning rate = 0.05")
    plt.ylabel("F1")
    plt.xlabel("#epoches")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    np.random.seed(0)
    np.set_printoptions(precision=12)
    main()
