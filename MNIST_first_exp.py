import csv,operator,sys,os
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from pandas.core.common import flatten

#sys.path.append('/home/mikail/uni/rmud/scikit-learn/mm-experiments/mnist')
def readFile(path):
    X = []
    Y = []

    f = open(path,'r')

    for row in f:
        entries = row.strip("\n").split(",")

        Y.append(int(entries[0])-1)
        x = [int(e) for e in entries[1:]]
        X.append(x)

    Y = np.array(Y)-min(Y)
    return np.array(X).astype(dtype=np.int32), Y

def get_nr_child_idx(clf):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    n_leaves_h = 0

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    # print("The binary tree structure has {n} nodes and has "
    #       "the following tree structure:\n".format(n=n_nodes))
    for i in range(n_nodes):
        if is_leaves[i]:
            # print("{space}node={node} is a leaf node.".format(
            #     space=node_depth[i] * "\t", node=i))
            n_leaves_h += 1
        #else:
            # print("{space}node={node} is a split node: "
            #       "go to node {left} if X[:, {feature}] <= {threshold} "
            #       "else to node {right}.".format(
            #           space=node_depth[i] * "\t",
            #           node=i,
            #           left=children_left[i],
            #           feature=feature[i],
            #           threshold=threshold[i],
            #           right=children_right[i]))
    return n_nodes - n_leaves_h

def main(argv):

    # MNIST
    #'''
    X_train,y_train = readFile("/home/mikail/uni/rmud/scikit-learn/mm-experiments/mnist/train.csv")
    X_test,y_test = readFile("/home/mikail/uni/rmud/scikit-learn/mm-experiments/mnist/test.csv")
    #'''
    nr_features = X_train.shape[1]

    # RF
    depths = [5, 10]
    estims = [5, 10]

    for dep in depths:
        for est in estims:
            # print("---- NEW RUN ----")
            # print("Estimators:{}, Depths:{}".format(est, dep))
            clf = RandomForestClassifier(max_depth=dep, n_estimators=est)
            clf = clf.fit(X_train, y_train)
            out = clf.predict(X_train)
            # print("Accuracy (train)", accuracy_score(y_train, out))

            # extract margins from tree
            # array with: margins, features, split values
            # margins_data = clf.tree_.get_margins(X_train)

            # bit flip injection data
            # print("bfi before", clf.tree_.bit_flip_injection)
            # clf.tree_.bit_flip_injection = 1
            # print("bfi after", clf.tree_.bit_flip_injection)
            # print("ber before", clf.tree_.bit_error_rate)
            # clf.tree_.bit_error_rate = np.array(0.01, dtype=np.float32)
            # print("ber after", clf.tree_.bit_error_rate)

            print("---------- BER TEST ----------")
            print("trees: {}, depth: {}".format(str(est), str(dep)))
            nr_points = 6
            ber_array = [0]
            ber_array.append([1*(10**(-nr_points+i)) for i in range(nr_points)])
            ber_array = list(flatten(ber_array))
            # print(ber_array)
            bers = np.array(ber_array, dtype=np.float32)
            reps = 5
            for ber in bers:
                for tree in clf.estimators_:
                    #print(tree)

                    # split value injection
                    # tree.tree_.bit_error_rate_split = ber
                    # tree.tree_.bit_flip_injection_split = 1

                    # feature index injection
                    # tree.tree_.bit_error_rate_featidx = ber
                    # tree.tree_.bit_flip_injection_featidx = 1
                    # tree.tree_.nr_feature_idx = np.floor(np.log2(nr_features)) + 1

                    # child indices injection
                    nr_ch_idx = get_nr_child_idx(tree)
                    nr_ch_idx *= 2
                    tree.tree_.nr_child_idx = np.floor(np.log2(nr_ch_idx)) + 1
                    tree.tree_.bit_error_rate_chidx = ber
                    tree.tree_.bit_flip_injection_chidx = 1

                acc_scores = []
                for rep in range(reps):
                    out = clf.predict(X_test)
                    acc_scores.append(accuracy_score(y_test, out))
                acc_scores_np = np.array(acc_scores)
                acc_mean = np.mean(acc_scores_np)
                acc_min = np.min(acc_scores_np)
                acc_max = np.max(acc_scores_np)
                # print("BER: {:.4f}, Accuracy: {:.4f} ({:.4f},{:.4f})".format(ber, acc_mean, acc_mean - acc_min, acc_max - acc_mean))
                print("{:.8f} {:.4f} {:.4f} {:.4f}".format(ber*100, (acc_mean)*100, (acc_max - acc_mean)*100, (acc_mean - acc_min)*100))

if __name__ == "__main__":
   main(sys.argv[1:])
