from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, average_precision_score

def tree_nrOfCorrectPredictionsDespiteWrongPath(X_train, y_train, X_test, y_test, depths, estims, bers, exp_path, dataset, print_trace=None):
    experiment_data = []
    for dep in depths:
        for est in estims:
            # create and train classifier
            tree = DecisionTreeClassifier(max_depth=dep)
            tree = tree.fit(X_train, y_train)
            for ber in bers:
                tree.tree_.bit_error_rate_split = ber
                tree.tree_.bit_flip_injection_split = 1

                counter_inputs = 0
                counter_correct_bwrong_path = 0
                for feature in X_test:
                    counter_inputs += 1
                    if print_trace is not None:
                        print("---new input---")
                    in1 = feature
                    # reshape because we use only one input sample
                    in1 = in1.reshape(1, -1)

                    print_trace = None

                    ### predict with errors
                    tree.tree_.bit_flip_injection_split = 1
                    # get path with errors
                    path_errors = tree.decision_path(in1, check_input=True)
                    if print_trace is not None:
                        print("MATRIX W/ ERRORS", path_errors)
                    # get label with errors
                    # get indices to non-zero data
                    rows, cols = path_errors.nonzero()
                    # extract error index
                    error_idx = cols[-1]
                    # get class
                    error_class_idx = np.argmax(tree.tree_.value[error_idx])
                    if print_trace is not None:
                        print("ERROR CLASS ->", error_class_idx)

                    ### predict without errors
                    tree.tree_.bit_flip_injection_split = 0
                    # out = tree.predict(in1)
                    # print("OUTPUT W/O ERROR", out)
                    # get path without errors
                    path_correct = tree.decision_path(in1, check_input=True)
                    if print_trace is not None:
                        print("MATRIX CORRECT", path_correct)
                    # get correct label:
                    # get indices to non-zero data
                    rows, cols = path_correct.nonzero()
                    # extract correct index
                    correct_idx = cols[-1]
                    # get class
                    correct_class_idx = np.argmax(tree.tree_.value[correct_idx])
                    if print_trace is not None:
                        print("CORRECT CLASS -> ", correct_class_idx)

                    comp = (path_correct!=path_errors).nnz==0
                    # print("ineq")
                    # print("the comparison", (sup!=sup2).nnz)
                    if print_trace is not None:
                        print("ARE THE PATHS EQUAL? ", comp)

                    # count cases with wrong paths, but different class idx
                    if (error_class_idx == correct_class_idx) and (comp == False):
                        counter_correct_bwrong_path += 1
                        if print_trace is not None:
                            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    if print_trace is not None:
                        print("------")
                if print_trace is not None:
                    print("---END---")

                # calc. ratio of correct predictions despite wrong path
                cc_wp = counter_correct_bwrong_path/counter_inputs
                if print_trace is not None:
                    print("Correct class despite wrong path ", cc_wp)

                experiment_data.append(
                    {
                        "model": "tree",
                        "dataset": dataset,
                        "depth": dep,
                        "perror_split": ber,
                        "cpdwp": cc_wp
                    }
                )
    return experiment_data

def tree_nrOfChangedPathsWithOneBF(X_train, y_train, X_test, y_test, depths, estims, bers, exp_path, dataset, print_trace=None):
    experiment_data = []
    for dep in depths:
        for est in estims:
            # create and train classifier
            tree = DecisionTreeClassifier(max_depth=dep)
            tree = tree.fit(X_train, y_train)
            for ber in bers:
                tree.tree_.bit_error_rate_split = ber
                tree.tree_.bit_flip_injection_split = 1
                tree.tree_.bf_occured_split = 0

                counter_inputs = 0
                counter_changed_paths = 0
                for feature in X_test:
                    counter_inputs += 1
                    if print_trace is not None:
                        print("---new input---")
                    in1 = feature
                    # reshape because we use only one input sample
                    in1 = in1.reshape(1, -1)

                    ### predict with errors
                    tree.tree_.bit_flip_injection_split = 1
                    # get path with errors
                    path_errors = tree.decision_path(in1, check_input=True)
                    if print_trace is not None:
                        print("MATRIX W/ ERRORS", path_errors)
                    # get label with errors
                    # get indices to non-zero data
                    rows, cols = path_errors.nonzero()
                    # extract error index
                    error_idx = cols[-1]
                    # get class
                    error_class_idx = np.argmax(tree.tree_.value[error_idx])
                    if print_trace is not None:
                        print("ERROR CLASS ->", error_class_idx)

                    ### predict without errors
                    tree.tree_.bit_flip_injection_split = 0
                    # out = tree.predict(in1)
                    # print("OUTPUT W/O ERROR", out)
                    # get path without errors
                    path_correct = tree.decision_path(in1, check_input=True)
                    if print_trace is not None:
                        print("MATRIX CORRECT", path_correct)
                    # get correct label:
                    # get indices to non-zero data
                    rows, cols = path_correct.nonzero()
                    # extract correct index
                    correct_idx = cols[-1]
                    # get class
                    correct_class_idx = np.argmax(tree.tree_.value[correct_idx])
                    if print_trace is not None:
                        print("CORRECT CLASS -> ", correct_class_idx)

                    comp = (path_correct!=path_errors).nnz==0
                    # print("ineq")
                    # print("the comparison", (sup!=sup2).nnz)
                    if print_trace is not None:
                        print("ARE THE PATHS EQUAL? ", comp)

                    # count cases with wrong paths, with one or more bfs
                    if (tree.tree_.bf_occured_split == 1) and (comp == False):
                        counter_changed_paths += 1
                        if print_trace is not None:
                            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    if print_trace is not None:
                        print("------")
                if print_trace is not None:
                    print("---END---")

                # calc. ratio of changes paths with one bit flip or more
                cp_onebf = counter_changed_paths/counter_inputs
                if print_trace is not None:
                    print("Changed path with one input ", cp_onebf)

                experiment_data.append(
                    {
                        "model": "tree",
                        "dataset": dataset,
                        "depth": dep,
                        "perror_split": ber,
                        "cp_onebf": cp_onebf
                    }
                )
    return experiment_data

def tree_PEs_estim(X_train, y_train, X_test, y_test, depths, estims, bers, exp_path, dataset, print_trace=None):
    experiment_data = []
    for dep in depths:
        for est in estims:
            # create and train classifier
            tree = DecisionTreeClassifier(max_depth=dep)
            tree = tree.fit(X_train, y_train)
            for ber in bers:
                tree.tree_.bit_error_rate_split = ber
                tree.tree_.bit_flip_injection_split = 1
                tree.tree_.bf_occured_split = 0

                counter_inputs = 0
                counter_changed_paths = 0

                counter_visited_nodes = 0
                counter_visited_nodes_with_errors = 0
                ratio_node_w_error_by_visited_nodes = 0
                ratio_node_w_error_and_correct_split_by_visited_nodes = 0
                for feature in X_test:
                    counter_inputs += 1
                    if print_trace is not None:
                        print("---new input---")
                    in1 = feature
                    # reshape because we use only one input sample
                    in1 = in1.reshape(1, -1)

                    ### predict with errors
                    tree.tree_.bit_flip_injection_split = 1
                    # get path with errors
                    path_errors = tree.decision_path(in1, check_input=True)
                    if print_trace is not None:
                        print("MATRIX W/ ERRORS", path_errors)
                    # get label with errors
                    # get indices to non-zero data
                    rows, cols = path_errors.nonzero()
                    # extract error index
                    error_idx = cols[-1]
                    # get class
                    error_class_idx = np.argmax(tree.tree_.value[error_idx])
                    if print_trace is not None:
                        print("ERROR CLASS ->", error_class_idx)
                    # print("Nr of nodes visited", tree.tree_.nr_nodes_visited)
                    # print("Nr of nodes visited w/ errors", tree.tree_.nr_nodes_visited_with_errors)
                    ratio_node_w_error_by_visited_nodes += (tree.tree_.nr_nodes_visited_with_errors / tree.tree_.nr_nodes_visited)
                    ratio_node_w_error_and_correct_split_by_visited_nodes += (tree.tree_.nr_nodes_visited_not_changing_path_despite_biterror / tree.tree_.nr_nodes_visited)


                    ### predict without errors
                    tree.tree_.bit_flip_injection_split = 0
                    # out = tree.predict(in1)
                    # print("OUTPUT W/O ERROR", out)
                    # get path without errors
                    path_correct = tree.decision_path(in1, check_input=True)
                    if print_trace is not None:
                        print("MATRIX CORRECT", path_correct)
                    # get correct label:
                    # get indices to non-zero data
                    rows, cols = path_correct.nonzero()
                    # extract correct index
                    correct_idx = cols[-1]
                    # get class
                    correct_class_idx = np.argmax(tree.tree_.value[correct_idx])
                    if print_trace is not None:
                        print("CORRECT CLASS -> ", correct_class_idx)

                    comp = (path_correct!=path_errors).nnz==0
                    # print("ineq")
                    # print("the comparison", (sup!=sup2).nnz)
                    if print_trace is not None:
                        print("ARE THE PATHS EQUAL? ", comp)

                    # count cases with wrong paths, with one or more bfs
                    if (tree.tree_.bf_occured_split == 1) and (comp == False):
                        counter_changed_paths += 1
                        if print_trace is not None:
                            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    if print_trace is not None:
                        print("------")
                if print_trace is not None:
                    print("---END---")

                # calc. ratio of changes paths with one bit flip or more
                cp_onebf = counter_changed_paths/counter_inputs
                if print_trace is not None:
                    print("Changed path with one input ", cp_onebf)

                experiment_data.append(
                    {
                        "model": "tree",
                        "dataset": dataset,
                        "depth": dep,
                        "perror_split": ber,
                        "cp_onebf": cp_onebf,
                        "nodes_visited_w_error_by_all_nodes_visited": ratio_node_w_error_by_visited_nodes / counter_inputs,
                        "ratio_node_w_error_and_correct_split_by_visited_nodes": ratio_node_w_error_and_correct_split_by_visited_nodes / counter_inputs
                    }
                )
    return experiment_data
