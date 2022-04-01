from datetime import datetime
import numpy as np
from pandas.core.common import flatten
import os
import json

def create_exp_folder(this_path):
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
        return exp_path

def store_exp_data_dict(exp_data):
    to_dump = dict()
    to_dump["results"] = exp_data
    return to_dump

def store_exp_data_write(to_dump_path, to_dump_data):
    with open(to_dump_path, 'a') as outfile:
        json.dump(to_dump_data, outfile)
        print ("Successfully stored results in %s" % to_dump_path)

def bit_error_rates_generator(p2exp):
    temp = [1, 2.5, 5.0, 7.5] # steps between powers of 10
    nr_points = p2exp # exponent to begin with, begin at 10**(-nr_points-1)
    ber_array = [0]
    ber_array.append([temp[i]*((10)**(-nr_points-1)) for i in range(len(temp))])
    ber_array = list(flatten(ber_array))
    rest_array = [1*(10**(-nr_points+i)) for i in range(nr_points)]
    for point in rest_array:
        for step in temp:
            ber_array.append(point*step)
    bers = bers[:-1]
    return bers

def quantize_data(data, q_range_bits):
    data = np.array(data)
    # get number of quantization levels
    q_range = 2**(q_range_bits) - 1
    # uniform quantization to unsigned integer, with q_range quantization levels
    # shift the data by the minumum value (negative or 0 here) so that all values are positive
    quantized = data - data.min()
    # multiply by the ratio of quantization range and data value range
    quantized = quantized * (q_range/(data.max() - data.min()))
    # round the result to nearest integer
    quantized = np.round(quantized)
    return quantized


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
