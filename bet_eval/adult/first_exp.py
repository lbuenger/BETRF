import csv,operator,sys,os
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from pandas.core.common import flatten
from datetime import datetime
import os
from sklearn.model_selection import train_test_split

# Preparations and configs
################################################################################
# paths to train and test
this_path = os.getcwd()

# train_path = this_path + "/dataset/train.csv"
# test_path = this_path + "/dataset/test.csv"
datset_path = this_path + "/dataset/adult.data"

# RF config
depths = [5, 10]
estims = [5, 10]

# configs for bit flip injection
split_inj = 0
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
def getFeatureVector(entries):
    x = []
    x.append(float(entries[0])) # age = continous

    workclass = [0 for i in range(8)]
    if entries[1] == "Private":
        workclass[0] = 1
    elif entries[1] == "Self-emp-not-inc":
        workclass[1] = 1
    elif entries[1] == "Self-emp-inc":
        workclass[2] = 1
    elif entries[1] == "Federal-gov":
        workclass[3] = 1
    elif entries[1] == "Local-gov":
        workclass[4] = 1
    elif entries[1] == "State-gov":
        workclass[5] = 1
    elif entries[1] == "Without-pay":
        workclass[6] = 1
    else: #Never-worked
        workclass[7] = 1
    x.extend(workclass)
    x.append(float(entries[2]))

    education = [0 for i in range(16)]
    if entries[3] == "Bachelors":
        education[0] = 1
    elif entries[3] == "Some-college":
        education[1] = 1
    elif entries[3] == "11th":
        education[2] = 1
    elif entries[3] == "HS-grad":
        education[3] = 1
    elif entries[3] == "Prof-school":
        education[4] = 1
    elif entries[3] == "Assoc-acdm":
        education[5] = 1
    elif entries[3] == "Assoc-voc":
        education[6] = 1
    elif entries[3] == "9th":
        education[7] = 1
    elif entries[3] == "7th-8th":
        education[8] = 1
    elif entries[3] == "12th":
        education[9] = 1
    elif entries[3] == "Masters":
        education[10] = 1
    elif entries[3] == "1st-4th":
        education[11] = 1
    elif entries[3] == "10th":
        education[12] = 1
    elif entries[3] == "Doctorate":
        education[13] = 1
    elif entries[3] == "5th-6th":
        education[14] = 1
    if entries[3] == "Preschool":
        education[15] = 1

    x.extend(education)
    x.append(float(entries[4]))

    marital = [0 for i in range(7)]
    if entries[5] == "Married-civ-spouse":
        marital[0] = 1
    elif entries[5] == "Divorced":
        marital[1] = 1
    elif entries[5] == "Never-married":
        marital[2] = 1
    elif entries[5] == "Separated":
        marital[3] = 1
    elif entries[5] == "Widowed":
        marital[4] = 1
    elif entries[5] == "Married-spouse-absent":
        marital[5] = 1
    else:
        marital[6] = 1
    x.extend(marital)

    occupation = [0 for i in range(14)]
    if entries[6] == "Tech-support":
        occupation[0] = 1
    elif entries[6] == "Craft-repair":
        occupation[1] = 1
    elif entries[6] == "Other-service":
        occupation[2] = 1
    elif entries[6] == "Sales":
        occupation[3] = 1
    elif entries[6] == "Exec-managerial":
        occupation[4] = 1
    elif entries[6] == "Prof-specialty":
        occupation[5] = 1
    elif entries[6] == "Handlers-cleaners":
        occupation[6] = 1
    elif entries[6] == "Machine-op-inspct":
        occupation[7] = 1
    elif entries[6] == "Adm-clerical":
        occupation[8] = 1
    elif entries[6] == "Farming-fishing":
        occupation[9] = 1
    elif entries[6] == "Transport-moving":
        occupation[10] = 1
    elif entries[6] == "Priv-house-serv":
        occupation[11] = 1
    elif entries[6] == "Protective-serv":
        occupation[12] = 1
    else:
        occupation[13] = 1
    x.extend(occupation)

    relationship = [0 for i in range(6)]
    if entries[7] == "Wife":
        relationship[0] = 1
    elif entries[7] == "Own-child":
        relationship[1] = 1
    elif entries[7] == "Husband":
        relationship[2] = 1
    elif entries[7] == "Not-in-family":
        relationship[3] = 1
    elif entries[7] == "Other-relative":
        relationship[4] = 1
    else:
        relationship[5] = 1
    x.extend(relationship)

    race = [0 for i in range(5)]
    if entries[8] == "White":
        race[0] = 1
    elif entries[8] == "Asian-Pac-Islander":
        race[1] = 1
    elif entries[8] == "Amer-Indian-Eskimo":
        race[2] = 1
    elif entries[8] == "Other":
        race[3] = 1
    else:
        race[4] = 1
    x.extend(race)

    if (entries[9] == "Male"):
        x.extend([1,0])
    else:
        x.extend([0,1])

    x.append(float(entries[10]))
    x.append(float(entries[11]))
    x.append(float(entries[12]))

    native = [0 for i in range(42)]
    if entries[14] == "United-States":
        native[1] = 1
    elif entries[14] == "Cambodia":
        native[2] = 1
    elif entries[14] == "England":
        native[3] = 1
    elif entries[14] == "Puerto-Rico":
        native[4] = 1
    elif entries[14] == "Canada":
        native[5] = 1
    elif entries[14] == "Germany":
        native[6] = 1
    elif entries[14] == "Outlying-US(Guam-USVI-etc)":
        native[7] = 1
    elif entries[14] == "India":
        native[8] = 1
    elif entries[14] == "Japan":
        native[9] = 1
    elif entries[14] == "Greece":
        native[10] = 1
    elif entries[14] == "South":
        native[11] = 1
    elif entries[14] == "China":
        native[12] = 1
    elif entries[14] == "Cuba":
        native[13] = 1
    elif entries[14] == "Iran":
        native[14] = 1
    elif entries[14] == "Honduras":
        native[15] = 1
    elif entries[14] == "Philippines":
        native[16] = 1
    elif entries[14] == "Italy":
        native[17] = 1
    elif entries[14] == "Poland":
        native[18] = 1
    elif entries[14] == "Jamaica":
        native[19] = 1
    elif entries[14] == "Vietnam":
        native[20] = 1
    elif entries[14] == "Mexico":
        native[21] = 1
    elif entries[14] == "Portugal":
        native[22] = 1
    elif entries[14] == "Ireland":
        native[23] = 1
    elif entries[14] == "France":
        native[24] = 1
    elif entries[14] == "Dominican-Republic":
        native[25] = 1
    elif entries[14] == "Laos":
        native[26] = 1
    elif entries[14] == "Ecuador":
        native[27] = 1
    elif entries[14] == "Taiwan":
        native[28] = 1
    elif entries[14] == "Haiti":
        native[29] = 1
    elif entries[14] == "Columbia":
        native[30] = 1
    elif entries[14] == "Hungary":
        native[31] = 1
    elif entries[14] == "Guatemala":
        native[32] = 1
    elif entries[14] == "Nicaragua":
        native[33] = 1
    elif entries[14] == "Scotland":
        native[34] = 1
    elif entries[14] == "Thailand":
        native[35] = 1
    elif entries[14] == "Yugoslavia":
        native[36] = 1
    elif entries[14] == "El-Salvador":
        native[37] = 1
    elif entries[14] == "Trinadad&Tobago":
        native[38] = 1
    elif entries[14] == "Peru":
        native[39] = 1
    elif entries[14] == "Hong":
        native[40] = 1
    else:
        native[41] = 1

    return x

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
    exp_data.write(datset_path+"\n")
    # exp_data.close()

    X = []
    Y = []

    f = open(datset_path)
    for row in f:
        if (len(row) > 1):
            entries = row.replace("\n", "").replace(" ", "").split(",")

            x = getFeatureVector(entries)

            if (entries[-1] == "<=50K"):
                y = 0
            else:
                y = 1

            Y.append(y)
            X.append(x)

    X = np.array(X).astype(dtype=np.int32)
    Y = np.array(Y)

    for iteration in range(0,3):
        if iteration == 0:
            split_inj = 1
            feature_idx_inj = 0
            child_idx_inj = 0
        if iteration == 1:
            split_inj = 0
            feature_idx_inj = 1
            child_idx_inj = 0
        if iteration == 2:
            split_inj = 0
            feature_idx_inj = 0
            child_idx_inj = 1

        exp_data.write("Threshold injection: "+str(split_inj)+"\n")
        exp_data.write("FeatureIdx injection: "+str(feature_idx_inj)+"\n")
        exp_data.write("ChildIdx injection: "+str(child_idx_inj)+"\n")

        rint = np.random.randint(low=1, high=100)
        X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=rint)

        # only for MNIST
        # X_train,y_train = readFile(train_path)
        # X_test,y_test = readFile(test_path)

        nr_features = X_train.shape[1]

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
                # exp_data = open(exp_path + "/results.txt", "a")
                exp_data.write("---------- BER TEST ----------\n")
                exp_data.write("trees: {}, depth: {}\n".format(str(est), str(dep)))
                # exp_data.close()
                for ber in bers:
                    for tree in clf.estimators_:
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
                    for rep in range(reps):
                        out = clf.predict(X_test)
                        acc_scores.append(accuracy_score(y_test, out))
                    acc_scores_np = np.array(acc_scores)
                    acc_mean = np.mean(acc_scores_np)
                    acc_min = np.min(acc_scores_np)
                    acc_max = np.max(acc_scores_np)
                    # print("BER: {:.4f}, Accuracy: {:.4f} ({:.4f},{:.4f})".format(ber, acc_mean, acc_mean - acc_min, acc_max - acc_mean))
                    # exp_data = open(exp_path + "/results.txt", "a")
                    exp_data.write("{:.8f} {:.4f} {:.4f} {:.4f}\n".format(ber*100, (acc_mean)*100, (acc_max - acc_mean)*100, (acc_mean - acc_min)*100))
                    # exp_data.close()
    exp_data.close()
if __name__ == "__main__":
   main(sys.argv[1:])
