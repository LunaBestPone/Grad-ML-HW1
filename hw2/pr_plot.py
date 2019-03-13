import sys, json
import numpy as np
import math
import matplotlib.pyplot as plt

def main():
    np.set_printoptions(precision=12)
    # simple arg check
    if len(sys.argv) != 3:
        print("wrong number of arguments")
        sys.exit(0)

    training_set_path = sys.argv[1]
    test_set_path = sys.argv[2]

    training_set_file = open(training_set_path,'r')
    test_set_file = open(test_set_path,'r')

    training_set = json.load(training_set_file)
    test_set = json.load(test_set_file)

    training_set_feature_matrix = np.array(training_set["data"])[:,:-1]
    test_set_feature_matrix = np.array(test_set["data"])[:,:-1]

    training_set_label_matrix = np.array(training_set["data"])[:,-1:]
    test_set_label_matrix = np.array(test_set["data"])[:,-1:]

    labels = training_set["metadata"]["features"][-1:][0][1]

    features = []
    for i in training_set["metadata"]["features"][:-1]:
        features.append(i[0])
    features_sub = []
    for i in training_set["metadata"]["features"][:-1]:
        features_sub.append(i[1])

    # set up P(Y)
    P_Y = np.zeros(len(labels))
    # fill up P_Y
    total_num_training_instances = training_set_label_matrix.shape[0]
    for idx,label in enumerate(labels):
        P_Y[idx] = np.divide(np.sum(training_set_label_matrix == label) + 1,total_num_training_instances + len(labels))

    # to access probability P(X=x|Y=y),P_X_on_Y[index of the feature name(category)][index of the value in all possible values for that feature][label]
    # P(X=arched|Y=malign) = P_X_on_Y[0][0][0] (first feature, first value of that feature, first label)
    P_X_on_Y = []
    for idx,feature_name in enumerate(features):
        curmatrix = np.zeros((len(features_sub[idx]),len(labels)))
        for jdx,feature_type in enumerate(features_sub[idx]):
            for kdx,label in enumerate(labels):
                # entries that satisfy y
                temp = training_set_label_matrix == label
                # numerator
                num_xy = np.sum(np.logical_and(temp.reshape(temp.shape[0]),training_set_feature_matrix[:,idx] == feature_type)) + 1
                # denominator
                num_y = np.sum(temp) + len(features_sub[idx])
                curmatrix[jdx,kdx] = np.divide(num_xy,num_y)
        P_X_on_Y.append(curmatrix)

    confidence = np.zeros(test_set_label_matrix.shape[0])



    for i in range(test_set_label_matrix.shape[0]):
        bot = 0
        top = np.zeros(len(labels))
        for j in range(len(labels)):
            mtop = 1.0
            mbottom = 1.0
            for k in range(len(features)):
                feature = test_set_feature_matrix[i][k]
                feature_index = features_sub[k].index(feature)
                mbottom *= P_X_on_Y[k][feature_index][j]
                mtop *= P_X_on_Y[k][feature_index][j]
            top[j] = P_Y[j] * mtop
            bot += P_Y[j] * mbottom

        predict_Plist = np.divide(top,bot)
        correct_index = np.argsort(-predict_Plist,kind="stable")[0]
        predict_p = round(predict_Plist[correct_index],12)
        predict_label = labels[correct_index]
        actual_label = test_set_label_matrix[i][0]
        confidence[i] = predict_Plist[0]

    pr = []
    re = []

    confidence_sorted_arg = np.argsort(-confidence,axis=0,kind="stable")
    sorted_confidence = confidence[confidence_sorted_arg]
    actual_pos = np.sum(test_set_label_matrix == labels[0],axis=0)
    tp = 0
    last_tp = 0
    for i in range(test_set_label_matrix.shape[0]):
        if i > 1 and sorted_confidence[i] != sorted_confidence[i-1]:
            predicted_pos = np.sum(confidence > sorted_confidence[i])
            precision = tp/predicted_pos
            pr.append(precision)
            recall = tp/actual_pos
            re.append(recall)
        if test_set_label_matrix[confidence_sorted_arg[i]] == labels[0]:
            tp += 1

    plt.plot(pr,re,label="naive bayes")

    # joint probability table
    pijy = []
    for i in features:
        pijy.append([0]*len(features))

    pij_y = []
    for i in features:
        pij_y.append([0]*len(features))


    for idx,featurei in enumerate(features):
        for jdx,featurej in enumerate(features):
                curmatrix = np.zeros((len(features_sub[idx]),len(features_sub[jdx]),len(labels)))
                curmatrix_cond = np.zeros((len(features_sub[idx]),len(features_sub[jdx]),len(labels)))
                for iidx,featureii in enumerate(features_sub[idx]):
                    for jjdx, featurejj in enumerate(features_sub[jdx]):
                        for kdx,label in enumerate(labels):
                            x1x2 = np.logical_and(training_set_feature_matrix[:,idx] == featureii,training_set_feature_matrix[:,jdx] == featurejj)
                            temp = training_set_label_matrix == label
                            xy = np.logical_and(temp.reshape(temp.shape[0]),x1x2)
                            numijx = np.sum(xy)
                            top = numijx + 1
                            #joint
                            bot = total_num_training_instances + len(features_sub[idx])*len(features_sub[jdx])*len(labels)
                            curmatrix[iidx][jjdx][kdx] = np.divide(top,bot)
                            #cond
                            bot2 = np.sum(temp) + len(features_sub[idx])*len(features_sub[jdx])
                            curmatrix_cond[iidx][jjdx][kdx] = np.divide(top,bot2)
                    pijy[idx][jdx] = curmatrix
                    pij_y[idx][jdx] = curmatrix_cond

    I = np.zeros((len(features),len(features)))
    for idx,featurei in enumerate(features):
        for jdx,featurej in enumerate(features):
            sum = 0
            for iidx,featureii in enumerate(features_sub[idx]):
                for jjdx, featurejj in enumerate(features_sub[jdx]):
                    for kdx,label in enumerate(labels):
                        sum += pijy[idx][jdx][iidx][jjdx][kdx] * math.log((pij_y[idx][jdx][iidx][jjdx][kdx]/(P_X_on_Y[idx][iidx][kdx] * P_X_on_Y[jdx][jjdx][kdx])),2)
            I[idx][jdx] = sum

    # find MST
    V = set(features)
    E = I
    Enew = []

    for i in range(len(features)):
        for j in range(len(features)):
            if i == j:
                E[i][j] = -float("inf")

    Vnew = set([features[0]])
    while (Vnew != V):
        max = -float("inf")
        maxi = 0
        maxj = 0
        for i in range(len(features)):
            for j in range(len(features)):
                if (E[i][j]>max and (features[i] in Vnew) and not(features[j] in Vnew)):
                    max = E[i][j]
                    maxi = i
                    maxj = j

        Vnew.add(features[maxj])
        Enew.append((maxi,maxj))


    root = features[0]
    parents = {}
    for feature in features:
        parents[feature] = []

    for f,t in Enew:
        parents[features[t]].append(features[f])

    for feature in features:
        parents[feature].append("class")

    #set up probs
    pxony2 = [0]*len(features)
    for idx,featurei in enumerate(features):
        if len(parents[featurei]) == 1:
            pxony2[idx] = P_X_on_Y[idx]
        else:
            jdx = features.index(parents[featurei][0])
            condmatrix = np.zeros((len(features_sub[idx]),len(features_sub[jdx]),len(labels)))
            for iidx,featureii in enumerate(features_sub[idx]):
                for jjdx, featurejj in enumerate(features_sub[jdx]):
                    for kdx,label in enumerate(labels):
                        x1x2 = np.logical_and(training_set_feature_matrix[:,idx] == featureii,training_set_feature_matrix[:,jdx] == featurejj)
                        temp = training_set_label_matrix == label
                        xy = np.logical_and(temp.reshape(temp.shape[0]),x1x2)
                        numijx = np.sum(xy)
                        top = numijx + 1
                        #cond
                        bot2 = np.sum(np.logical_and(training_set_feature_matrix[:,jdx] == featurejj, temp.reshape(temp.shape[0]))) + len(features_sub[idx])
                        condmatrix[iidx][jjdx][kdx] = np.divide(top,bot2)
            pxony2[idx] = condmatrix

    #calculation
    for i in range(test_set_label_matrix.shape[0]):
        bot = 0
        top = np.zeros(len(labels))
        for j in range(len(labels)):
            mtop = 1.0
            mbottom = 1.0
            for k in range(len(features)):
                feature = test_set_feature_matrix[i][k]
                feature_index = features_sub[k].index(feature)
                if(len(parents[features[k]])==1):
                    mbottom *= P_X_on_Y[k][feature_index][j]
                    mtop *= P_X_on_Y[k][feature_index][j]
                else:
                    pfeature = parents[features[k]][0]
                    pfeature_i = features.index(pfeature)
                    pfeatureval = test_set_feature_matrix[i][pfeature_i]
                    pfeature_index = features_sub[pfeature_i].index(pfeatureval)
                    mbottom *= pxony2[k][feature_index][pfeature_index][j]
                    mtop *= pxony2[k][feature_index][pfeature_index][j]
            top[j] = P_Y[j] * mtop
            bot += P_Y[j] * mbottom

        predict_Plist = np.divide(top,bot)
        correct_index = np.argsort(-predict_Plist,kind="stable")[0]
        predict_p = round(predict_Plist[correct_index],12)
        predict_label = labels[correct_index]
        actual_label = test_set_label_matrix[i][0]
        confidence[i] = predict_Plist[0]

    pr = []
    re = []

    confidence_sorted_arg = np.argsort(-confidence,axis=0,kind="stable")
    sorted_confidence = confidence[confidence_sorted_arg]
    actual_pos = np.sum(test_set_label_matrix == labels[0],axis=0)
    tp = 0
    for i in range(test_set_label_matrix.shape[0]):
        if i > 1 and sorted_confidence[i] != sorted_confidence[i-1]:
            predicted_pos = np.sum(confidence > sorted_confidence[i])
            precision = tp/predicted_pos
            pr.append(precision)
            recall = tp/actual_pos
            re.append(recall)
        if test_set_label_matrix[confidence_sorted_arg[i]] == labels[0]:
            tp += 1

    plt.plot(pr,re,label="TAN")
    plt.ylabel("precision")
    plt.xlabel("recall")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
