import numpy as np
from pandas.core.common import flatten
from scipy.stats import norm
import matplotlib.pyplot as plt

# class for storing split value, an array of deviated split values (with all possible combinations of bit flips), and an array of values storing the number of flipped bits of the first array
class BFvals:
    def __init__(self, value, bits):
        # the value to inject in
        self.value = value
        # number of bits in the value
        self.bits = bits
        # array of deviated split values (with all possible combinations of bit flips)
        self.injected_vals = None
        # array of values storing the number of flipped bits of the first array
        self.nr_flipped_bits_in_injected_val = None

        # create values to inject with xor
        injection_vals = np.array([i for i in range(0, 2**(bits))], dtype=np.uint8)
        # print("INJ vals", injection_vals)

        # expand for unpackbits
        injection_vals_exp = np.expand_dims(injection_vals, axis=1)
        # print("INJ2 vals", injection_vals_exp)

        # popcount to calculate the number of flipped bits
        setbits = np.sum(np.unpackbits(injection_vals_exp, axis=1), axis=1)
        # print("SETBITS", setbits)

        # prepare value to inject in
        value = np.array([value], dtype=np.uint8)

        # create all possible deviated values with bit flip injection
        injected_vals = injection_vals ^ value

        # assign arrays to object
        self.injected_vals = injected_vals
        self.nr_flipped_bits_in_injected_val = setbits
        # print("XOR inj. vals", xor_injection_vals)

        # print("SHAPES", xor_injection_vals.shape, setbits.shape)

    # function to return tuple of deviated value and nr of flipped bits
    def return_tuple(self, index):
        tuple = (self.injected_vals[index], self.nr_flipped_bits_in_injected_val[index])
        return tuple

# 8 bit only for now
def tree_pe2(splits_list, features_list, bers, dep):
    nr_bits_split = 7
    # node data is in form list of lists
    # print("max: ", splits_list)
    # splits_list_np = np.array(splits_list, dtype=np.uint8)
    # features_list_np = np.array(features_list, dtype=np.uint8)
    print("In tree_pe2")
    p_Sqs_for_all_paths = []

    counter = 0
    max_samples = 10000
    # iterate over paths
    for ber in bers:
        p_Sqs_for_all_paths = []
        counter = 0
        for (split_path, feature_path) in zip(splits_list, features_list):
            if counter < max_samples:
                counter += 1
                # print(counter)
                p_Sqs_for_one_path = []
                # iterate over split and feature values in one path
                for (split, feature) in zip(split_path, feature_path):
                    # print("Comparison: ", (feature, split))
                    bfvals = BFvals(split, nr_bits_split)

                    p_Sq = 0
                    # iterate over number of bits in split value representation
                    for b in range(1, nr_bits_split+1):
                        # print("b:", b)
                        # compute the probability of "b" bit flips in the split value
                        p_b = ber**(b) * (1-ber)**(nr_bits_split-b)
                        # nr_comparisons w/ bit errors
                        comp_total_cases = 0
                        # nr_comparisons w/ wrong result
                        comp_wrong_cases = 0
                        # compare feature value with split value
                        comp_correct = (feature <= split)
                        # compare a deviated split value with feature
                        for inj_idx in range(0, 2**(bfvals.bits)):
                            # only when the number of flipped bits is equal to b
                            tup = bfvals.return_tuple(inj_idx)
                            split_err = tup[0]
                            split_err_fb = tup[1]
                            if split_err_fb == b:
                                comp_total_cases += 1
                                comp_error = (feature <= split_err)
                                comp_comp = np.equal(comp_correct, comp_error)
                                # if comparison w/ errors is wrong
                                if comp_comp == 0:
                                    comp_wrong_cases += 1
                        mistakes_div_cases = comp_wrong_cases / comp_total_cases
                        p_Sq += (p_b * mistakes_div_cases)
                        # print("mistakes/cases", mistakes_div_cases)
                        # print("---")
                    # done with a node, append to path buffer
                    p_Sqs_for_one_path.append(p_Sq)
                    # print(p_Sqs_for_one_path)
                    # print("--")
                    # print("ber: {}, p_Sq: {} ".format(ber, p_Sq))
                # done with one path, append to buffer for all paths
                p_Sqs_for_all_paths.append(p_Sqs_for_one_path)
        # p_Sqs_for_all_paths_np = np.array(p_Sqs_for_all_paths)
        # print("Done ", p_Sqs_for_all_paths_np.shape)
        # print("Done ", p_Sqs_for_all_paths_np)

        # calculate correct prediction probabilities
        prob_for_correct_path = []
        for path_probs in p_Sqs_for_all_paths:
            correct_path_prob = 1
            for node_prob in path_probs:
                correct_path_prob *= (1 - node_prob)
            # print("prob for correct path: ", correct_path_prob)
            prob_for_correct_path.append(correct_path_prob)

        prob_for_correct_path_np = np.array(prob_for_correct_path)
        avg = np.mean(prob_for_correct_path_np)
        mini = np.min(prob_for_correct_path_np)
        maxi = np.max(prob_for_correct_path_np)
        # print("correct path probs: ", prob_for_correct_path_np)
        print("-> ber: {} -- {}, ({},{}): ".format(ber, avg, mini, maxi))

        # mu, std = norm.fit(prob_for_correct_path_np)
        # s = np.random.normal(mu, std, 10)
        # plt.hist(prob_for_correct_path_np, bins=50, density=True, alpha=0.6, color='g')
        # xmin, xmax = plt.xlim()
        # x = np.linspace(xmin, xmax, 1000)
        # p = norm.pdf(x, mu, std)
        # plt.plot(x, p, 'k', linewidth=2)
        # title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
        # plt.title(title)
        # plt.savefig("out_d_{}_ber_{}.png".format(dep, ber), format="png")
