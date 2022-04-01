from sklearn.model_selection import train_test_split
from loadData import readFileMNIST, readFileAdult, readFileSensorless, readFileWinequality, readFileSpamBase, readFileWearable
from sklearn.datasets import load_iris, fetch_olivetti_faces, fetch_covtype
from Utils import quantize_data
import numpy as np

def getData(dataset, this_path, nr_bits_split, nr_bits_feature, random_state):
    X_train, y_train, X_test, y_test = None, None, None, None

    if dataset == "MNIST":
        # nr_bits_split = 8
        # nr_bits_feature = 8
        dataset_train_path = "/data/mnist/train.csv"
        dataset_test_path = "/data/mnist/test.csv"
        train_path = this_path + dataset_train_path
        test_path = this_path + dataset_test_path
        X_train, y_train = readFileMNIST(train_path)
        X_test, y_test = readFileMNIST(test_path)

    if dataset == "IRIS":
        # nr_bits_split = 7
        # nr_bits_feature = 7
        dataset_train_path = "/sklearn"
        dataset_test_path = "/sklearn"
        train_path = this_path + dataset_train_path
        test_path = this_path + dataset_test_path
        iris = load_iris()
        X, y = iris.data, iris.target
        X *= 10
        X = X.astype(np.uint8)
        # rint = np.random.randint(low=1, high=100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)

    if dataset == "ADULT":
        # nr_bits_split = 8 # int, use 32 for fp
        # nr_bits_feature = 8 # int, use 32 for fp
        dataset_path = "data/adult/adult.data"
        X, y = readFileAdult(dataset_path)
        # rint = np.random.randint(low=1, high=100)
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=random_state)
        # comment out quantization, if it is not desired
        X_train = quantize_data(X_train, nr_bits_feature)
        X_test = quantize_data(X_test, nr_bits_feature)

    if dataset == "SENSORLESS":
        # nr_bits_split = 32 # floating point
        # nr_bits_feature = 32 # floating point
        dataset_path = "data/sensorless-drive/Sensorless_drive_diagnosis.txt"
        X, y = readFileSensorless(dataset_path)
        # rint = np.random.randint(low=1, high=100)
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=random_state)

    if dataset == "WINEQUALITY":
        # nr_bits_split = 32 # floating point
        # nr_bits_feature = 32 # floating point
        dataset_path = "data/wine-quality/"
        X, y = readFileWinequality(dataset_path)
        # rint = np.random.randint(low=1, high=100)
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=random_state)

    if dataset == "OLIVETTI":
        # nr_bits_split = 8
        # nr_bits_feature = 8
        dataset_train_path = "/sklearn"
        dataset_test_path = "/sklearn"
        train_path = this_path + dataset_train_path
        test_path = this_path + dataset_test_path
        X, y = fetch_olivetti_faces(shuffle=True, random_state=random_state, download_if_missing=True, return_X_y=True)
        X = np.array(X*255).astype(np.uint8) # use unsigned ints
        # rint = np.random.randint(low=1, high=100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)

    if dataset == "COVTYPE":
        # nr_bits_split = 8
        # nr_bits_feature = 8
        dataset_train_path = "/sklearn"
        dataset_test_path = "/sklearn"
        train_path = this_path + dataset_train_path
        test_path = this_path + dataset_test_path
        X, y = fetch_covtype("data/covtype/", shuffle=True, random_state=random_state, download_if_missing=True, return_X_y=True)
        X = np.array(X).astype(np.uint8) # use unsigned ints
        # rint = np.random.randint(low=1, high=100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)

    if dataset == "SPAMBASE":
        # nr_bits_split = 16
        # nr_bits_feature = 16
        dataset_path = "data/spambase/spambase.data"
        X, y = readFileSpamBase(dataset_path)
        # print("ma", X.max(), X.min())
        # X = np.array(X).astype(np.uint8)
        # rint = np.random.randint(low=1, high=100)
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=random_state)
        X_train = quantize_data(X_train, nr_bits_feature)
        X_test = quantize_data(X_test, nr_bits_feature)

    if dataset == "WEARABLE":
        # nr_bits_split = 16
        # nr_bits_feature = 16
        dataset_path = "data/wearable/dataset.csv"
        X, y = readFileWearable(dataset_path)
        # print("ma", X.max(), X.min())
        # X = np.array(X).astype(np.uint8)
        # rint = np.random.randint(low=1, high=100)
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=random_state)
        X_train = quantize_data(X_train, nr_bits_feature)
        X_test = quantize_data(X_test, nr_bits_feature)

    return X_train, y_train, X_test, y_test
