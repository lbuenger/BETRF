# libary imports
import csv,operator,sys,os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from pandas.core.common import flatten
from datetime import datetime
import os
import joblib

# own file imports
from Utils import create_exp_folder, store_exp_data_dict, store_exp_data_write, bit_error_rates_generator
from loadData import readFileMNIST, readFileAdult, readFileSensorless
from pathEvals import tree_nrOfCorrectPredictionsDespiteWrongPath, tree_nrOfChangedPathsWithOneBF, tree_PEs_estim
from bfi_evaluation import bfi_tree, bfi_forest

def main():
    ### Preparations and configs
    # paths to train and test
    this_path = os.getcwd()

    # command line arguments, use argparse here later
    dataset = "SENSORLESS"

    # DT/RF configs
    DT_RF = "RF" # DT or RF
    depth = 5 # DT/RF depth (single value for DT, list for RFs)
    estims = 5 # number of DTs in RF (does not matter for DT)
    split_inj = 1 # activate split value injection with 1
    feature_inj = 0 # activate feature value injection with 1
    nr_bits_split = None # nr of bits in split value, it is set below when dataset is loaded
    int_split = 0 # whether to use integer split
    nr_bits_feature = None # nr of bits in feature value, it is set below when dataset is loaded
    feature_inj = 0 # activate feature value injection with 1
    feature_idx_inj = 0 # activate feature idx injection with 1
    child_idx_inj = 0 # activate child idx injection with 1
    reps = 5 # how many times to evaluate for one bit error rate
    # p2exp = 6 # error rates for evaluation start at 2^(-p2exp)
    # bers = bit_error_rates_generator(p2exp)
    bers = [0, 0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 1]
    export_accuracy = 1 # 1 if accuracy list for a bit error rate should be exported as .npy, else None
    all_data = []

    # read data
    train_path = ""
    test_path = ""
    X_train, y_train, X_test, y_test = None, None, None, None

    if dataset == "MNIST":
        nr_bits_split = 8
        nr_bits_feature = 8
        dataset_train_path = "/mnist/dataset/train.csv"
        dataset_test_path = "/mnist/dataset/test.csv"
        train_path = this_path + dataset_train_path
        test_path = this_path + dataset_test_path
        X_train, y_train = readFileMNIST(train_path)
        X_test, y_test = readFileMNIST(test_path)

    if dataset == "IRIS":
        nr_bits_split = 7
        nr_bits_feature = 7
        dataset_train_path = "/sklearn"
        dataset_test_path = "/sklearn"
        train_path = this_path + dataset_train_path
        test_path = this_path + dataset_test_path
        iris = load_iris()
        X, y = iris.data, iris.target
        X *= 10
        X = X.astype(np.uint8)
        # rint = np.random.randint(low=1, high=100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    if dataset == "ADULT":
        nr_bits_split = 32 # floating point
        nr_bits_feature = 32 # floating point
        dataset_path = "adult/dataset/adult.data"
        X, y = readFileAdult(dataset_path)
        # rint = np.random.randint(low=1, high=100)
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    if dataset == "SENSORLESS":
        nr_bits_split = 32 # floating point
        nr_bits_feature = 32 # floating point
        dataset_path = "sensorless-drive/dataset/Sensorless_drive_diagnosis.txt"
        X, y = readFileSensorless(dataset_path)
        # rint = np.random.randint(low=1, high=100)
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    # TODO: loop over models here, and use multiprocessing
    # train or load tree / forest
    # clf = DecisionTreeClassifier(max_depth=depth)
    # model = clf.fit(X_train, y_train)
    # joblib.dump(model, "DT{}_{}.pkl".format(depth, dataset), compress=9)
    # model = joblib.load('PREL_MODELS/MNIST/DT10_MNIST.pkl')
    clf = RandomForestClassifier(max_depth=depth, n_estimators=estims)
    model = clf.fit(X_train, y_train)

    # create experiment folder and return the path to it
    exp_path = create_exp_folder(this_path)

    # create data file to store experiment results
    exp_data = open(exp_path + "/results.txt", "a")
    exp_data.write(train_path+"\n")
    exp_data.write(test_path+"\n")
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
        "export_accuracy": export_accuracy
        }

    # call evaluation function
    # bfi_tree(expdata_dict)
    bfi_forest(expdata_dict)

    # dump experiment settings to file, but first remove unserializable elements
    keys_to_remove = ["model", "X_train", "X_test","y_train","y_test"]
    for key in keys_to_remove:
        expdata_dict.pop(key)
    to_dump_data = expdata_dict
    to_dump_path = exp_path + "/results.txt"
    store_exp_data_write(to_dump_path, to_dump_data)

    # visualize model (for tree)
    # fig = plt.figure(figsize=(25,20))
    # _ = tree.plot_tree(clf,
    #                    feature_names=iris.feature_names,
    #                    class_names=iris.target_names,
    #                    filled=True)
    # fig.savefig("DT{}_{}.png".format(depth, dataset))

if __name__ == '__main__':
    main()
