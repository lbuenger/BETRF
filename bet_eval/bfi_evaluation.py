from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, average_precision_score

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

def bfi_tree(exp_dict):

    print("a", exp_dict["dataset_name"])
    # write experiment data to file
    nr_features = exp_dict["X_train"].shape[1]

    # get data from dictionary
    tree = exp_dict["model"]
    X_test = exp_dict["X_test"]
    y_test = exp_dict["y_test"]
    reps = exp_dict["reps"]
    bers = exp_dict["bers"]
    dataset_name = exp_dict["dataset_name"]
    export_accuracy = exp_dict["export_accuracy"]

    # split
    split_inj = exp_dict["split_inj"]
    int_split = exp_dict["int_split"]
    nr_bits_split = exp_dict["nr_bits_split"]

    # feature
    feature_inj = exp_dict["feature_inj"]
    feature_idx_inj = exp_dict["feature_idx_inj"]

    # child
    child_idx_inj = exp_dict["child_idx_inj"]

    exp_path = exp_dict["experiment_path"]
    # exp_data = exp_dict["experiment_data"]

    exp_data = open(exp_path + "/results.txt", "a")
    exp_data.write("--- BER TEST ---\n")
    estims = exp_dict["estims"]
    depth = exp_dict["depth"]
    exp_data.write("trees: {}, depth: {}, reps: {}, dataset: {}\n".format(estims, depth, reps, dataset_name))
    # exp_data.close()
    for ber in bers:
        exp_data = open(exp_path + "/results.txt", "a")
        # reset configs
        tree.tree_.bit_flip_injection_split = 0
        tree.tree_.bit_flip_injection_featval = 0
        tree.tree_.bit_flip_injection_featidx = 0
        tree.tree_.bit_flip_injection_chidx = 0

        # split value injection
        if split_inj == 1:
            tree.tree_.bit_error_rate_split = ber
            tree.tree_.bit_flip_injection_split = split_inj
            tree.tree_.int_rounding_for_thresholds = int_split
            tree.tree_.int_threshold_bits = nr_bits_split

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
            out = tree.predict(X_test)
            acc_scores.append(accuracy_score(y_test, out))
        acc_scores_np = np.array(acc_scores)
        # print("ACC scores", acc_scores_np)
        acc_mean = np.mean(acc_scores_np)
        # print("means:", acc_mean)
        acc_min = np.min(acc_scores_np)
        acc_max = np.max(acc_scores_np)
        # print("BER: {:.4f}, Accuracy: {:.4f} ({:.4f},{:.4f})".format(ber, acc_mean, acc_mean - acc_min, acc_max - acc_mean))
        # exp_data = open(exp_path + "/results.txt", "a")
        exp_data.write("{:.8f} {:.4f} {:.4f} {:.4f}\n".format(ber*100, (acc_mean)*100, (acc_max - acc_mean)*100, (acc_mean - acc_min)*100))

        # dump exp data for each error rate
        if export_accuracy is not None:
            filename = "ber_{}.npy".format(ber*100)
            with open(filename, 'wb') as f:
            	np.save(f, acc_scores_np)
        # exp_data.close()
        # print("{:.8f} {:.4f} {:.4f} {:.4f}\n".format(ber*100, (acc_mean)*100, (acc_max - acc_mean)*100, (acc_mean - acc_min)*100))
        exp_data.close()
