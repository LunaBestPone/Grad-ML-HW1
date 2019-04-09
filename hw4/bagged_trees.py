import numpy as np
import sys,json
import DecisionTree as dt

def main():
    # arg check
    if len(sys.argv) != 5:
        print("wrong number of arguments")
        sys.exit(0)
    # passing args
    numtrees = int(sys.argv[1])
    maxdepth = int(sys.argv[2])
    training_set_path = sys.argv[3]
    test_set_path = sys.argv[4]

    # get data
    train = json.load(open(training_set_path, 'r'))
    train_meta = train['metadata']['features']
    class_list = train_meta[-1][1]
    train_data = np.array(train['data'])
    test = json.load(open(test_set_path, 'r'))
    test_data = np.array(test['data'])
    train_x = train_data[:,:-1]
    train_y = train_data[:,-1]
    test_x = test_data[:,:-1]
    test_y = test_data[:,-1]


    # list of trees
    trees = []
    for i in range(numtrees):
        trees.append(dt.DecisionTree())
    # table of training sample indices
    NTtable = np.zeros((train_data.shape[0],numtrees),dtype=int)
    for i in range(numtrees):
        NTtable[:,i] = np.random.choice(train_data.shape[0],train_data.shape[0],replace=True)
    for i in NTtable:
        out = ""
        for j in i[:-1]:
            out += str(j)+","
        out += str(i[-1])
        print(out)
    print("")
    # training
    for i in range(numtrees):
        trees[i].fit(train_x[NTtable[:,i],:], train_y[NTtable[:,i]], train_meta, max_depth=maxdepth)
    # table of predicted y
    predicted_y = trees[0].predict(test_x,prob=False)
    for i in range(1,numtrees):
        predicted_y = np.column_stack((predicted_y,trees[i].predict(test_x,prob=False)))
    predicted_y_p = []
    for i in range(numtrees):
        predicted_y_p.append(trees[i].predict(test_x,prob=True))
    combined_y = []
    for i in range(test_x.shape[0]):
        probs = np.zeros((numtrees,len(class_list)))
        for j in range(numtrees):
            curtreeprob = predicted_y_p[j][i]
            probs[j,:] = curtreeprob
        avg = np.mean(probs,axis=0)
        index = np.argmax(avg)
        combined_class = class_list[index]
        combined_y.append(combined_class)
    combined_y = np.array(combined_y)
    predicted_y = np.column_stack((predicted_y,combined_y))
    output_table = np.column_stack((predicted_y,test_y))
    for idx,i in enumerate(predicted_y):
        out = ""
        for j in i:
            out += str(j)+","
        out += str(test_y[idx])
        print(out)
    print("")
    acc = np.sum(output_table[:,-2] == output_table[:,-1])/test_y.shape[0]
    print(acc)


if __name__ == "__main__":
    np.random.seed(0)
    main()
