import csv,operator,sys,os
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

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
    # IRIS
    '''
    X_train, y_train = load_iris(return_X_y=True)
    clf = tree.DecisionTreeClassifier(max_depth=10)
    clf = clf.fit(X_train, y_train)
    out = clf.predict(X_train)
    print("Accuracy", accuracy_score(y_train, out))
    '''

    # MNIST
    #'''
    X_train,y_train = readFile("/home/mikail/uni/rmud/scikit-learn/mm-experiments/mnist/train.csv")
    X_test,y_test = readFile("/home/mikail/uni/rmud/scikit-learn/mm-experiments/mnist/test.csv")
    #'''


    # DT
    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf = clf.fit(X_train, y_train)
    out = clf.predict(X_train)
    print("---------- TRAINING ENDED ----------\n")
    print("Accuracy (train)", accuracy_score(y_train, out))

    # extract margins from tree
    # array with: margins, features, split values
    margins_data = clf.tree_.get_margins(X_train)

    # bit flip injection data
    print("bfi split before", clf.tree_.bit_flip_injection_split)
    print("bfi ch_idx before", clf.tree_.bit_flip_injection_chidx)
    clf.tree_.bit_flip_injection_split = 1
    clf.tree_.bit_flip_injection_chidx = 1
    print("bfi split after", clf.tree_.bit_flip_injection_split)
    print("bfi ch_idx after", clf.tree_.bit_flip_injection_chidx)
    print("ber split before", clf.tree_.bit_error_rate_split)
    print("ber ch_idx before", clf.tree_.bit_error_rate_chidx)
    clf.tree_.bit_error_rate_split = np.array(0.01, dtype=np.float32)
    clf.tree_.bit_error_rate_chidx = np.array(0.098, dtype=np.float32)
    print("ber split after", clf.tree_.bit_error_rate_split)
    print("ber ch_idx after", clf.tree_.bit_error_rate_chidx)

    ### Find out number of nodes that are not lead nodes
    ### For determining the required number of bits for addressing child indices
    ### per tree
    nr_ch_idx = get_nr_child_idx(clf)
    nr_ch_idx *= 2
    print("nr child index before", clf.tree_.nr_child_idx)
    clf.tree_.nr_child_idx = np.floor(np.log2(nr_ch_idx)) + 1
    print("nr child index after", clf.tree_.nr_child_idx)

    print("---------- BER TEST ----------")

    bers = np.array([i*0.0000001 for i in range(10)], dtype=np.float32)
    reps = 5
    for ber in bers:
        aborted_counter = 0
        clf.tree_.bit_error_rate_split = ber
        clf.tree_.bit_error_rate_chidx = ber
        acc_scores = []
        for rep in range(reps):
            out = clf.predict(X_test)
            acc_scores.append(accuracy_score(y_test, out))
            aborted_counter += clf.tree_.aborted
            # reset abouted counters in trees
            clf.tree_.aborted = 0
        acc_scores_np = np.array(acc_scores)
        acc_mean = np.mean(acc_scores_np)
        acc_min = np.min(acc_scores_np)
        acc_max = np.max(acc_scores_np)
        # print("BER: {:.4f}, Accuracy: {:.4f} ({:.4f},{:.4f})".format(ber, acc_mean, acc_mean - acc_min, acc_max - acc_mean))
        print("{:.4f} {:.4f} {:.4f} {:.4f} (aborted:{})".format(ber*100, (acc_mean)*100, (acc_mean - acc_min)*100, (acc_max - acc_mean)*100, aborted_counter))
        # print("{:.4f} {:.4f} {:.4f} {:.4f} (aborted:{})".format(ber*100, (acc_mean)*100, (acc_mean - acc_min)*100, (acc_max - acc_mean)*100, aborted_counter))
        # break

    '''
    # RF
    depths = [5, 10, 15, 20]
    estims = [5, 10, 15, 20]

    for dep in depths:
        for est in estims:
            print("---- NEW RUN ----")
            print("Estimators:{}, Depths:{}".format(est, dep))
            clf = RandomForestClassifier(max_depth=dep, n_estimators=est)
            clf = clf.fit(X_train, y_train)
            out = clf.predict(X_train)
            print("Accuracy (train)", accuracy_score(y_train, out))

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

            bers = np.array([i*0.001 for i in range(100)], dtype=np.float32)
            reps = 5
            for ber in bers:
                for tree in clf.estimators_:
                    #print(tree)
                    tree.tree_.bit_error_rate = ber
                    tree.tree_.bit_flip_injection = 1
                    #print(clf.estimators_[0].tree_.bit_error_rate)
                acc_scores = []
                for rep in range(reps):
                    out = clf.predict(X_test)
                    acc_scores.append(accuracy_score(y_test, out))
                acc_scores_np = np.array(acc_scores)
                acc_mean = np.mean(acc_scores_np)
                acc_min = np.min(acc_scores_np)
                acc_max = np.max(acc_scores_np)
                # print("BER: {:.4f}, Accuracy: {:.4f} ({:.4f},{:.4f})".format(ber, acc_mean, acc_mean - acc_min, acc_max - acc_mean))
                print("{:.4f} {:.4f} {:.4f} {:.4f}".format(ber*100, (acc_mean)*100, (acc_mean - acc_min)*100, (acc_max - acc_mean)*100))
    '''
    '''
    margins = margins_data[0]
    features = margins_data[1]
    splitvalues = margins_data[2]

    # TODO: this can probably be done with axis = 1 in one call
    mean_m = np.mean(margins)
    min_m = np.min(margins)
    max_m = np.max(margins)

    mean_f = np.mean(features)
    min_f = np.min(features)
    max_f = np.max(features)

    mean_sv = np.mean(splitvalues)
    min_sv = np.min(splitvalues)
    max_sv = np.max(splitvalues)
    '''
    # print("Margins -> Mean: {}, min: {}, max:{}".format(mean_m, min_m, max_m))
    # print("Features -> Mean: {}, min: {}, max:{}".format(mean_f, min_f, max_f))
    # print("Split values -> Mean: {}, min: {}, max:{}".format(mean_sv, min_sv, max_sv))

    #print(clf.tree_.value)
    #X_train = np.array(X_train, dtype=np.float32)
    #print(clf.tree_.apply(X_train))

    '''
    fig1, ax1 = plt.subplots()
    ax1.set_title('Margins')
    ax1.boxplot(margins)

    fig2, ax2 = plt.subplots()
    ax2.set_title('Features')
    ax2.boxplot(features)

    fig3, ax3 = plt.subplots()
    ax3.set_title('Split Values')
    ax3.boxplot(splitvalues)
    '''
    # plt.show()
    # plt.pause(3)
    # plt.close()


if __name__ == "__main__":
   main(sys.argv[1:])
