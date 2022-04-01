# libary imports
import csv,operator,sys,os
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from scipy.stats import norm
from pandas.core.common import flatten
from datetime import datetime
import os
import joblib

# own file imports
from Utils import create_exp_folder, store_exp_data_dict, store_exp_data_write, bit_error_rates_generator
from pathEvals import tree_nrOfCorrectPredictionsDespiteWrongPath, tree_nrOfChangedPathsWithOneBF, tree_PEs_estim
from bfi_evaluation import bfi_tree, bfi_forest
from prepareExpData import getData

def main():
    ### Preparations and configs
    # paths to train and test
    this_path = os.getcwd()

    # command line arguments, use argparse here later
    dataset = "WEARABLE"

    # DT/RF configs
    DT_RF = "RF" # DT or RF (needs to be correctly specified when loading a model)
    depth = 5 # DT/RF depth (single value for DT, list for RFs)
    estims = 5 # number of DTs in RF (does not matter for DT)
    split_inj = 1 # activate split value injection with 1
    feature_inj = 0 # activate feature value injection with 1
    nr_bits_split = 32 # nr of bits in split value
    nr_bits_feature = 32 # nr of bits in feature value
    int_split = 0 # whether to use integer split
    feature_inj = 0 # activate feature value injection with 1
    feature_idx_inj = 0 # activate feature idx injection with 1
    child_idx_inj = 0 # activate child idx injection with 1
    reps = 5 # how many times to evaluate for one bit error rate
    # p2exp = 6 # error rates for evaluation start at 2^(-p2exp)
    # bers = bit_error_rates_generator(p2exp)
    # bers = [0, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
    bers = [0, 0.0001, 0.001]
    export_accuracy = 1 # 1 if accuracy list for a bit error rate should be exported as .npy, else None
    all_data = []
    random_state = 42 #np.random.randint(low=1, high=100)
    store_model = None
    load_model = None
    # load_model = "experiments/MNIST_RF_T5_D5/RF_D5_T5_MNIST.pkl"
    # load_model = "DT5_MNIST.pkl"
    # load_model = "RF_D5_T5_MNIST.pkl"
    plot_histogram = None # plots histogram of input data (useful for quantization)
    true_majority = 1

    # read data
    X_train, y_train, X_test, y_test = getData(dataset, this_path, nr_bits_split, nr_bits_feature, random_state)

    # create experiment folder and return the path to it
    exp_path = create_exp_folder(this_path)

    # TODO: loop over models here, and use multiprocessing
    # train or load tree / forest
    if load_model is not None:
        model = joblib.load(load_model)
    else:
        if DT_RF == "DT":
            clf = DecisionTreeClassifier(max_depth=depth)
            model = clf.fit(X_train, y_train)
            if store_model is not None:
                joblib.dump(model, exp_path+f"/D{depth}_{dataset}.pkl", compress=9)

        if DT_RF == "RF":
            clf = RandomForestClassifier(max_depth=depth, n_estimators=estims)
            model = clf.fit(X_train, y_train)
            if store_model is not None:
                joblib.dump(model, exp_path+f"/D{depth}_T{estims}_{dataset}.pkl", compress=9)

    # create data file to store experiment results
    exp_data = open(exp_path + "/results.txt", "a")
    exp_data.close()

    # dictionary for experiment data
    expdata_dict = {
        "DT_RF": DT_RF,
        "model": model,
        "depth": depth,
        "estims": estims,
        "dataset_name": dataset,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "experiment_path": exp_path,
        "split_inj": split_inj,
        "int_split": int_split,
        "feature_inj": feature_inj,
        "nr_bits_split": nr_bits_split,
        "nr_bits_feature": nr_bits_feature,
        "feature_idx_inj": feature_idx_inj,
        "child_idx_inj": child_idx_inj,
        "reps": reps,
        "bers": bers,
        "export_accuracy": export_accuracy,
        "true_majority": true_majority
        }

    # call evaluation function
    if DT_RF == "DT":
        bfi_tree(expdata_dict)
    if DT_RF == "RF":
        bfi_forest(expdata_dict)

    # dump experiment settings to file, but first remove unserializable elements
    keys_to_remove = ["model", "X_train", "X_test","y_train","y_test"]
    for key in keys_to_remove:
        expdata_dict.pop(key)
    to_dump_data = expdata_dict
    to_dump_path = exp_path + "/results.txt"
    # TODO convert "bers" to a python array before dumping
    store_exp_data_write(to_dump_path, to_dump_data)

    if plot_histogram is not None:
        # print("Xtrain", X_train)

        #plot histogram of values
        mu, std = norm.fit(X_train.flatten())
        s = np.random.normal(mu, std, 100)
        plt.hist(X_train.flatten(), bins=50, color='g')
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 1000)
        # p = norm.pdf(x, mu, std)
        # plt.plot(x, p, 'k', linewidth=2)
        plt.semilogy()
        title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
        plt.title(title)
        plt.savefig("input_distr.pdf", format="pdf")

        min = np.abs(X_train).min()
        max = np.abs(X_train).max()
        print(f"min: {min}, max: {max}")

    # visualize model (for tree)
    # fig = plt.figure(figsize=(25,20))
    # _ = tree.plot_tree(clf,
    #                    feature_names=iris.feature_names,
    #                    class_names=iris.target_names,
    #                    filled=True)
    # fig.savefig("DT{}_{}.png".format(depth, dataset))

if __name__ == '__main__':
    main()
