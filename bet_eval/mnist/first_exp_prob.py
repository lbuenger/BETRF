import csv,operator,sys,os
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from pandas.core.common import flatten
from datetime import datetime
import os

# Preparations and configs
################################################################################
# paths to train and test
this_path = os.getcwd()

train_path = this_path + "/dataset/train.csv"
test_path = this_path + "/dataset/test.csv"

# RF config
depths = [5]
estims = [1]

# configs for bit flip injection
split_inj = 1
feature_idx_inj = 0
child_idx_inj = 0
reps = 5 # how many times to evaluate for one bit error rate

# config for x axis in plot
temp = [1, 2.5, 5.0, 7.5] # steps between powers of 10
nr_points = 6 # exponent to begin with, begin at 10**(-nr_points-1)
ber_array = [0]
ber_array.append([temp[i]*((10)**(-nr_points-1)) for i in range(len(temp))])
ber_array = list(flatten(ber_array))
rest_array = [1*(10**(-nr_points+i)) for i in range(nr_points)]
for point in rest_array:
    for step in temp:
        ber_array.append(point*step)
bers = np.array(ber_array, dtype=np.float32)
bers = bers[:-1]
bers = [0]

# create experiment folder
exp_path = ""
access_rights = 0o755
this_path = os.getcwd()
exp_path += this_path+"/experiments/"+"results-"+datetime.now().strftime('%d-%m-%Y-%H:%M:%S')
try:
    os.makedirs(exp_path, access_rights, exist_ok=False)
except OSError:
    print ("Creation of the directory %s failed" % exp_path)
else:
    print ("Successfully created the directory %s" % exp_path)
################################################################################

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
    # write experiment data to file
    exp_data = open(exp_path + "/results.txt", "a")
    exp_data.write(train_path+"\n")
    exp_data.write(test_path+"\n")
    # exp_data.close()

    # only for MNIST
    X_train,y_train = readFile(train_path)
    X_test,y_test = readFile(test_path)

    for iteration in range(0,1):
        if iteration == 0:
            split_inj = 1
            feature_idx_inj = 0
            child_idx_inj = 0

        nr_features = X_train.shape[1]

        exp_data.write("Threshold injection: "+str(split_inj)+"\n")
        exp_data.write("FeatureIdx injection: "+str(feature_idx_inj)+"\n")
        exp_data.write("ChildIdx injection: "+str(child_idx_inj)+"\n")

        for dep in depths:
            for est in estims:
                # print("---- NEW RUN ----")
                # print("Estimators:{}, Depths:{}".format(est, dep))
                tree = DecisionTreeClassifier(max_depth=dep)
                tree = tree.fit(X_train, y_train)
                # out = clf.predict(X_train)
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
                # exp_data = open(exp_path + "/results.txt", "a")
                exp_data.write("---------- BER TEST ----------\n")
                exp_data.write("trees: {}, depth: {}\n".format(str(est), str(dep)))
                # exp_data.close()
                for ber in bers:
                    # for tree in clf.estimators_:
                    #print(tree)

                    # reset configs
                    tree.tree_.bit_error_rate_split = 0
                    tree.tree_.bit_flip_injection_split = 0
                    tree.tree_.bit_error_rate_featidx = 0
                    tree.tree_.bit_flip_injection_featidx = 0
                    tree.tree_.bit_error_rate_chidx = 0
                    tree.tree_.bit_flip_injection_chidx = 0

                    # split value injection
                    if split_inj == 1:
                        tree.tree_.bit_error_rate_split = ber
                        tree.tree_.bit_flip_injection_split = 1

                    # feature index injection
                    if feature_idx_inj == 1:
                        tree.tree_.bit_error_rate_featidx = ber
                        tree.tree_.bit_flip_injection_featidx = 1
                        tree.tree_.nr_feature_idx = np.floor(np.log2(nr_features)) + 1

                    # child indices injection
                    if child_idx_inj == 1:
                        nr_ch_idx = get_nr_child_idx(tree)
                        nr_ch_idx *= 2
                        tree.tree_.nr_child_idx = np.floor(np.log2(nr_ch_idx)) + 1
                        tree.tree_.bit_error_rate_chidx = ber
                        tree.tree_.bit_flip_injection_chidx = 1

                    acc_scores = []
                    # sup = tree.decision_path(X_test, check_input=True)
                    # print("matrixX", sup.shape)
                    # print("matrixX", sup)

                    tree.tree_.bit_flip_injection_split = 1
                    tree.tree_.bit_error_rate_split = 0.025

                    print_trace = None

                    counter_inputs = 0
                    counter_correct_bwrong_path = 0
                    for feature in X_test:
                        counter_inputs += 1
                        print("---new input---")
                        in1 = feature
                        # reshape because we use only one input sample
                        in1 = in1.reshape(1, -1)

                        ### predict with errors
                        tree.tree_.bit_flip_injection_split = 1
                        # get path with errors
                        path_errors = tree.decision_path(in1, check_input=True)
                        print("MATRIX W/ ERRORS", path_errors)
                        # get label with errors
                        # get indices to non-zero data
                        rows, cols = path_errors.nonzero()
                        # extract error index
                        error_idx = cols[-1]
                        # get class
                        error_class_idx = np.argmax(tree.tree_.value[error_idx])
                        print("ERROR CLASS ->", error_class_idx)

                        ### predict without errors
                        tree.tree_.bit_flip_injection_split = 0
                        # out = tree.predict(in1)
                        # print("OUTPUT W/O ERROR", out)
                        # get path without errors
                        path_correct = tree.decision_path(in1, check_input=True)
                        print("MATRIX CORRECT", path_correct)
                        # get correct label:
                        # get indices to non-zero data
                        rows, cols = path_correct.nonzero()
                        # extract correct index
                        correct_idx = cols[-1]
                        # get class
                        correct_class_idx = np.argmax(tree.tree_.value[correct_idx])
                        print("CORRECT CLASS -> ", correct_class_idx)

                        comp = (path_correct!=path_errors).nnz==0
                        # print("ineq")
                        # print("the comparison", (sup!=sup2).nnz)
                        print("ARE THE PATHS EQUAL? ", comp)

                        # count cases with wrong paths, but different class idx
                        if (error_class_idx == correct_class_idx) and (comp == False):
                            counter_correct_bwrong_path += 1
                            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

                        print("------")
                    print("---END---")

                    # calc. ratio of correct predictions despite wrong path
                    cc_wp = counter_correct_bwrong_path/counter_inputs
                    print("Correct class despite wrong path ", cc_wp)
                    '''
                    for rep in range(reps):
                        out = tree.predict(X_test)
                        acc_scores.append(accuracy_score(y_test, out))
                    acc_scores_np = np.array(acc_scores)
                    acc_mean = np.mean(acc_scores_np)
                    acc_min = np.min(acc_scores_np)
                    acc_max = np.max(acc_scores_np)
                    # print("BER: {:.4f}, Accuracy: {:.4f} ({:.4f},{:.4f})".format(ber, acc_mean, acc_mean - acc_min, acc_max - acc_mean))
                    # exp_data = open(exp_path + "/results.txt", "a")
                    exp_data.write("{:.8f} {:.4f} {:.4f} {:.4f}\n".format(ber*100, (acc_mean)*100, (acc_max - acc_mean)*100, (acc_mean - acc_min)*100))
                    # exp_data.close()
                    # print("{:.8f} {:.4f} {:.4f} {:.4f}\n".format(ber*100, (acc_mean)*100, (acc_max - acc_mean)*100, (acc_mean - acc_min)*100))
                    '''
    exp_data.close()
if __name__ == "__main__":
   main(sys.argv[1:])
