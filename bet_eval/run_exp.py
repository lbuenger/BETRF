import csv,operator,sys,os
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from pandas.core.common import flatten
from datetime import datetime
import os

from Utils import create_exp_folder, store_exp_data_dict, store_exp_data_write, bit_error_rates_generator
from loadData import readFileMNIST
from pathEvals import tree_nrOfCorrectPredictionsDespiteWrongPath, tree_nrOfChangedPathsWithOneBF, tree_PEs_estim

def main():
    ### Preparations and configs
    # paths to train and test
    this_path = os.getcwd()

    # command line arguments, use argparse here later
    dataset = "MNIST"

    # read data
    train_path = ""
    test_path = ""
    X_train, y_train, X_test, y_test = None, None, None, None
    if dataset == "MNIST":
        dataset_train_path = "/mnist/dataset/train.csv"
        dataset_test_path = "/mnist/dataset/test.csv"
        train_path = this_path + dataset_train_path
        test_path = this_path + dataset_test_path
        X_train, y_train = readFileMNIST(train_path)
        X_test, y_test = readFileMNIST(test_path)

    # create experiment folder and return the path to it
    exp_path = create_exp_folder(this_path)

    # create data file to store experiment results
    exp_data = open(exp_path + "/results.txt", "a")
    exp_data.write(train_path+"\n")
    exp_data.write(test_path+"\n")
    exp_data.close()

    # DT/RF configs
    depths = [5] # DT/RF depths
    estims = [1] # number of DTs in RF
    split_inj = 0 # activate split value injection
    feature_idx_inj = 0 # activate feature idx injection
    child_idx_inj = 0 # activate child idx injection
    reps = 5 # how many times to evaluate for one bit error rate
    # p2exp = 6 # error rates for evaluation start at 2^(-p2exp)
    # bers = bit_error_rates_generator(p2exp)
    bers = [0.0001, 0.001, 0.01, 0.1, 0.25]
    all_data = []

    # call evaluation function
    # exp_data_results = tree_nrOfCorrectPredictionsDespiteWrongPath(X_train, y_train, X_test, y_test, depths, estims, bers, exp_path, dataset)
    # exp_data_results = tree_nrOfChangedPathsWithOneBF(X_train, y_train, X_test, y_test, depths, estims, bers, exp_path, dataset)

    exp_data_results = tree_PEs_estim(X_train, y_train, X_test, y_test, depths, estims, bers, exp_path, dataset)

    to_dump_data = exp_data_results
    to_dump_path = exp_path + "/results.txt"
    store_exp_data_write(to_dump_path, to_dump_data)

if __name__ == '__main__':
    main()
