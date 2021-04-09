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
    print("Accuracy (train)", accuracy_score(y_train, out))

    # extract margins from tree
    # array with: margins, features, split values
    margins_data = clf.tree_.get_margins(X_train)

    # bit flip injection data
    print("bfi before", clf.tree_.bit_flip_injection)
    clf.tree_.bit_flip_injection = 1
    print("bfi after", clf.tree_.bit_flip_injection)
    print("ber before", clf.tree_.bit_error_rate)
    clf.tree_.bit_error_rate = np.array(0.01, dtype=np.float32)
    print("ber after", clf.tree_.bit_error_rate)

    print("---------- BER TEST ----------")

    bers = np.array([i*0.001 for i in range(100)], dtype=np.float32)
    reps = 5
    for ber in bers:
        clf.tree_.bit_error_rate = ber
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
        break

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
