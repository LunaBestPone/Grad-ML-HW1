# "DecisionTree.py"
# David Merrell
# 2019-03-19
# 
# Implementation of a simple ID3 decision tree classifier
# for CS760 Spring 2019, HW4

import numpy as np

########################################
class DecisionTree:
    """
    Simple ID3 decision tree. 
    methods:
        fit(X, y, metadata, depth=None)

        predict(X, prob=False) 

        __str__() 
    """

    ####################################
    def __init__(self):
        """
        Do-nothing constructor
        """

        self._metadata = [] 
        self._feat_to_ind = {}
        self._model = {}
        
        return


    #####################################
    def fit(self, X, y, metadata, max_depth=None, instance_weights=None):
        """
        Train the decision tree on features X and 
        labels y. 
        
        "metadata" is a list of the features (and label), 
        with their possible values. 
        (note: this corresponds to the "features" list
        in our JSON files.)

        By default, the tree's max depth is unlimited--the
        algorithm continues until all splits are uninformative.

        instance_weights allows the user to provide relative
        importances of the training instances. It must
        be a nonnegative 1D array-like object with length equal to the number
        of training instances.
        """

        if max_depth is None:
            max_depth = np.inf

        if instance_weights is None:
            instance_weights = np.ones(X.shape[0])

        assert (instance_weights.shape == (X.shape[0],)), "instance weights are wrong shape: {}".format(instance_weights.shape)
        instance_weights = np.array(instance_weights)
        instance_weights = instance_weights / instance_weights.shape[0]

        self._metadata = metadata
        self._feat_to_ind = {feat[0]:i for i, feat in enumerate(metadata)}
        self._model = _rec_buildtree(X, y, metadata, self._feat_to_ind, 
                                     0, max_depth, instance_weights)

        return


    ####################################
    def predict(self, X, prob=False, pseudocount=None):
        """
        Predict the labels for test feature vectors X.
        If "prob" is False, return the modal class.
        If "prob" is True, return class probabilities
        computed with the given "pseudocount".

        Default value for pseudocount is 1/training set size.
        """

        labels = []

        if pseudocount is None:
            pseudocount = 1.0 / self._model["__counts__"].sum()

        # Loop through the test set
        for i in range(X.shape[0]):

            node = self._model
            test_instance = X[i,:]

            while "__children__" in node.keys():

                splitvar = node["__splitvar__"]

                # Numeric split
                if 'numeric' in self._metadata[self._feat_to_ind[splitvar]]:
                    if test_instance[self._feat_to_ind[splitvar]].astype(np.float) >= node["__splitval__"]:
                        node = node["__children__"]["__>=__"]
                    else:
                        node = node["__children__"]["__<__"]
                
                # Categorical split
                else:
                    # We're at a categorical variable split:
                    node = node["__children__"][test_instance[self._feat_to_ind[splitvar]]]
              
            # We have arrived at a leaf.
            # probabilistic classification:
            if prob:
                labels.append( (node["__weights__"] + pseudocount) / (node["__weights__"].sum() + pseudocount*len(node["__weights__"])) )
            # modal classification:
            else:
                maxind = np.argmax(node["__weights__"])
                labels.append( self._metadata[-1][1][maxind] )

        return np.array(labels) 



    ####################################
    def __str__(self):

        def rec_tostr(node, h):

            string = ""
            if "__children__" not in node.keys():
                #string += "\n" + "|   "*h + node["__weights__"].__str__()
                string += "\n" + "|   "*h + node["__counts__"].__str__()
                return string

            else:  
                splval = ""
                if node["__splitval__"] is not None:
                    splval = node["__splitval__"]

                string += "\n" + "|   "*h + str(node["__splitvar__"])
                
                for k, child in sorted(node["__children__"].items()):
                    string += "\n" + "|   "*h + "| {}{} {}".format(k, 
                                                                   splval, 
                                                                   rec_tostr(child, 
                                                                             h+1))

            return string

 
        string = rec_tostr(self._model, 0) 
        return string[1:]


########################################
# helper functions
########################################


def _rec_buildtree(X, y, metadata, feat_to_ind, depth, max_depth,
                   instance_weights): 
    """
    Recursive decision tree builder function.

    @param X (2D numpy array-like): the feature data
    @param y (1D numpy array-like): the labels
    @param metadata (list): metadata describing the columns of data
                            (last entry describes the classes)
    @param depth (int): the maximum depth of the decision tree.
    @param instance_weights (1D numpy array-like): weights over training instances.

    Is called by the DecisionTree.fit(...) method.
    """

    instance_weights = instance_weights / np.sum(instance_weights)

    # Build the current node; record its count for each label.
    node = {}
    node["__weights__"] = np.array([instance_weights[y == lbl].sum() for lbl in metadata[-1][1]])
    node["__counts__"] = np.array([(y == lbl).sum() for lbl in metadata[-1][1]])

    # Base cases...
    # max-depth and pure node cases:
    if depth >= max_depth or (1.0 - np.max(node["__weights__"]) < 1e-12):
        return node

    # Look at the splits and get the best one
    splitvar, splitval = _get_best_split(X, y, metadata, instance_weights)

    # Catch no-split and no-informative-split cases
    if splitvar is None:
        return node
    
    node["__splitvar__"] = splitvar
    node["__splitval__"] = splitval 
    node["__children__"] = {}

    # Recursive cases...
    # numeric split:
    if metadata[feat_to_ind[splitvar]][1] == 'numeric':
        ge_inds = (X[:,feat_to_ind[splitvar]].astype(np.float) >= splitval)
        node["__children__"]["__>=__"] = _rec_buildtree(X[ge_inds,:], 
                                                    y[ge_inds],
                                                    metadata,
                                                    feat_to_ind,
                                                    depth+1,
                                                    max_depth,
                                                    instance_weights[ge_inds])
        node["__children__"]["__<__"] = _rec_buildtree(X[~ge_inds,:], 
                                                   y[~ge_inds], 
                                                   metadata,
                                                   feat_to_ind,
                                                   depth+1,
                                                   max_depth,
                                                   instance_weights[~ge_inds])
    # categorical split:
    else:
        for val in metadata[feat_to_ind[splitvar]][1]:
            eq_inds = X[:, feat_to_ind[splitvar]] == val
            node["__children__"][val] = _rec_buildtree(X[eq_inds,:],
                                                   y[eq_inds],
                                                   metadata,
                                                   feat_to_ind,
                                                   depth+1,
                                                   max_depth,
                                                   instance_weights[eq_inds])
    
    return node 


def _get_best_split(X, y, metadata, instance_weights):
    """
    Get the split with greatest mutual information.
    """

    splitvar = None
    splitval = ""
    maxinfo = 0

    for i, feat in enumerate(metadata[:-1]):
        
        # If numeric feature:
        if feat[1] == 'numeric':

            Xcol = X[:,i].astype(np.float)
            
            # Sort the numeric values 
            # (and their corresponding y's)
            srt_inds = np.argsort(X[:,i], kind='mergesort')
            srt_feat = Xcol[srt_inds]
            srt_y = y[srt_inds]
            srt_weights = instance_weights[srt_inds]
               
            # Get all of the midpoints between consecutive instances
            # that undergo a class change:
            midpoints = [ 0.5*(x + srt_feat[j+1]) for j,x in enumerate(srt_feat[:-1]) if y[j] != y[j+1]]
            midpoints = sorted(list(set(midpoints)))

            # For each midpoint: evaluate the mutual information
            # associated with its split  
            for mp in midpoints:
                mi = mutual_info( (Xcol >= mp).astype(np.int), y, p=instance_weights)
                if mi > maxinfo:
                    maxinfo = mi
                    splitvar = feat[0] 
                    splitval = mp
        
        # if a categorical feature:
        else:
            mi = mutual_info(X[:,i], y, p=instance_weights)
            if mi > maxinfo:
                maxinfo = mi
                splitvar = feat[0]
                splitval = ""
        
    return splitvar, splitval


################################################
def mutual_info(x, y, p=None):
    """
    Return the mutual information between
    columns x and y. Columns are assumed to 
    be discrete-valued (i.e., with entries from
    discrete domains x_dom and y_dom).

    "p" allows the user to specify weights over
    the entries of x. We assume p sums to 1.
    """

    if p is None:
        p = np.ones(x.shape[0]) / x.shape[0]

    xvals = np.unique(x)
    yvals = np.unique(y)

    # Get the (weighted) x counts
    xval_to_ind = {xv:i for i, xv in enumerate(xvals)}
    x_inds = np.array( [xval_to_ind[v] for v in x], dtype=int)
    x_enc = np.zeros((x.shape[0], xvals.shape[0]))
    x_enc[np.arange(x.shape[0],dtype=int),x_inds] = p 
    p_x = np.sum(x_enc, axis=0, keepdims=False)

    # Get the (weighted) y counts
    yval_to_ind = {yv:i for i, yv in enumerate(yvals)}
    y_inds = np.array( [yval_to_ind[v] for v in y], dtype=int)
    y_enc = np.zeros((y.shape[0], yvals.shape[0]))
    y_enc[np.arange(y.shape[0],dtype=int),y_inds] = p 
    p_y = np.sum(y_enc, axis=0, keepdims=False)
    
    # Get the (weighted) x,y joint counts
    xy_counts = np.zeros((x.shape[0], xvals.shape[0], yvals.shape[0]))
    xy_counts[np.arange(x.shape[0],dtype=np.int), x_inds, y_inds] = p 
    p_xy = np.sum(xy_counts, axis=0, keepdims=False)

    # Entropy of x (in bits)
    lp_x = np.log2(p_x, where=(p_x != 0.0))
    h_x = -np.dot(p_x, lp_x)

    # Compute conditional entropy H(x|y):
    h_xgy = np.zeros(yvals.shape[0])
    for j, _ in enumerate(yvals):
        p_xgy = p_xy[:,j] / p_y[j]
        lp_xgy = np.log2(p_xgy, where=(p_xgy != 0.0))
        h_xgy[j] = -1.0*np.dot(p_xgy, lp_xgy)

    # MI = entropy - conditional entropy
    return h_x - 1.0*np.dot(p_y, h_xgy)


if __name__=="__main__":

    import sys
    import json
    args = sys.argv

    depth = int(args[1])
    train_file = args[2]
    test_file = args[3]

    # Example usage of the DecisionTree class: 
    mydt = DecisionTree()
    train = json.load(open(train_file,"r"))
    train_X = np.array(train['data'])
    train_y = train_X[:,-1]
    train_X = train_X[:,:-1]
    meta = train['metadata']['features']
    mydt.fit(train_X, train_y, meta, max_depth=depth)

    print(mydt)
    
    test = json.load(open(test_file,"r"))
    test_X = np.array(test['data'])
    test_y = test_X[:,-1]
    test_X = test_X[:,:-1]
    
    preds = mydt.predict(test_X, prob=False)
   
    print( (preds == test_y).sum() / preds.shape[0] )
