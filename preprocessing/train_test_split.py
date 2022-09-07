import os
import random
import sys

import pandas as pd
import enum
import numpy as np
from itertools import combinations_with_replacement, product
from utils import ncr
from preprocessing.testsplit_config import *
from scripts.data_cleaning.pairwise_similarity import get_rna_sequences
import math
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
import seaborn as sns


class TrainRegime(enum.Enum):
    RANDOM = 1
    SIMILARITY_K_FOLDS = 2


class TrainTestSplit:
    """
    class that receives a path to a database and then splits the data in that database to train, validation and test
    sets
    Has two main partition regimes that can be added to:
    1) Random train test split using existing libraries
    2) Conditioned split using a similarity matrix.
    """

    def __init__(self, db_path: str, split_regime: int, sim_matrix: str, train_percent: float, k: int,
                 threshold: float, output_db: str, RNA_list_path = "/cs/usr/punims/Desktop/punims-dinaLab/Databases/Database_1_MG_RNA/rna_map.npy",
                 fasta_db = "/cs/usr/punims/Desktop/punims-dinaLab/Databases/Database_1_MG_RNA_fasta"):
        """
        :param output_db: output directory where we can save extra memory.
        :param train_percent: percentage of data that should be in he training set.
        :param k: number of folds
        :param db_path: path to database
        :param split_regime: what function to use to split the data
        :param sim_matrix: similarity matrix for one type of the splits.
        """

        self.db_path = db_path
        self.split_regime = split_regime
        self.sim_matrix = pd.read_csv(sim_matrix)
        self.train_percent = train_percent
        self.test_percent = 1.0 - train_percent
        self.k = k
        self.threshold = threshold
        self.output_db = output_db

        self.RNA_list_path = RNA_list_path
        self.fasta_db = fasta_db
        self.pvalue = 0.01
        self.cutoff = 0.9
        self.bin_size = 5
        self.total_sim_thresh = 0.56
        self.banned_sequences = set()

        temp_sim_matrix = self.sim_matrix.values + self.sim_matrix.values.T
        temp_sim_matrix[[np.arange(temp_sim_matrix.shape[0])]*2] = 1
        self.plot_sim_matrix(temp_sim_matrix, "Similarity Matrix before filtering")
        self.drop_similar_samples(self.total_sim_thresh)

        # turn upper triangle similarity matrix values to full symmetric matrix 
        self.complete_sim_matrix = self.sim_matrix.values + self.sim_matrix.values.T
        self.complete_sim_matrix[[np.arange(self.complete_sim_matrix.shape[0])]*2] = 1
        self.plot_sim_matrix(self.complete_sim_matrix, "Similarity Matrix after Filtering")

    def plot_sim_matrix(self, sim_matrix, title):
        """
        given a similarity matrix plots it using consistent color scheme.
        @return:
        """

        im = plt.imshow(sim_matrix, cmap='viridis')
        plt.title(title)
        plt.colorbar(im)
        plt.show()



    def calculate_connected_components(self):
        adjacency_matrix = np.copy(self.complete_sim_matrix)
        adjacency_matrix[np.where(self.complete_sim_matrix > self.threshold)] = 1
        adjacency_matrix[np.where(self.complete_sim_matrix <= self.threshold)] = 0
        num_of_components = connected_components(adjacency_matrix)
        print(f"number of connected components is {num_of_components} with a threshold of {self.threshold}")

    def split_selection(self):
        """
        split regime selection using enum class
        :return:
        """
        my_regime = TrainRegime(self.split_regime)
        if my_regime == TrainRegime.RANDOM:
            return self.random_split()
        elif my_regime == TrainRegime.SIMILARITY_K_FOLDS:
            return self.similarity_k_folds()

    def random_split(self):
        """
        regular split train split by tensorflow
        :return:
        """
        pass

    def similarity_k_folds(self):
        """
        Choose a percentage of the data for train and test.
        Add at random that percentage first to the test set then iteratively check using the similarity
        matrix what samples aren't too similar to those in the test set. Whatever is too similar can be used
        for validation.
        do this until we have 'k' valid sets that aren't too small in the training set.
        to save memory, save the k-folds as text files that will write where each sample should be used.
        :return:
        """

        ROW_IDX = 0
        FILE_NAME_IDX = 1

        columns = self.sim_matrix.columns
        pdbfiles_and_chains = {key: val for key, val in enumerate(columns)}
        valid_iterations = 0

        seeds = self.get_seed_generator()
        statistics_map, bins, bin_map = self.get_statistics(self.pvalue, self.cutoff)
        while self.k > valid_iterations:
            if len(seeds) == 0:
                print(f"Not enough validation sets found, found a total of {valid_iterations} valid sets")
                break
            for seed in seeds:
                train_set, validation_set, test_set = self.split_train_val_test(seed, pdbfiles_and_chains, statistics_map, bins, bin_map)
                seeds.remove(seed)
                train_size_target = 0.55 * self.train_percent * len(pdbfiles_and_chains)
                if len(train_set) > train_size_target:
                    print(f"splitting with seed {seed} was successful, sizes: training:{len(train_set)}, validation:{len(validation_set)}, test:{len(test_set)}")
                    # atleast 75% from the training percent made it in:
                    valid_iterations += 1
                    with open(os.path.join(self.output_db, f"training_list_seed_{seed}_cutoff_{self.cutoff}_ban_{self.total_sim_thresh}.txt"), 'w') as file:
                        file.write(f"seed used is {seed}\n")
                        file.write("Training Examples: \n")
                        for training_example in train_set:
                            file.write(training_example[FILE_NAME_IDX]+"\n")
                        file.write("Test Examples: \n")
                        for test_example in test_set:
                            file.write(test_example[FILE_NAME_IDX]+"\n")
                        file.write("Validation Examples: \n")
                        for valid_example in validation_set:
                            file.write(valid_example[FILE_NAME_IDX]+"\n")
                    break

                else:
                    print(f"train list was not large enough for seed {seed} length of train_list was {len(train_set)} while target is {train_size_target}")


    def split_train_val_test(self, seed, pdbfiles_and_chains, statistics_map, bins, bin_map):
        """
        helper function for kfolds to split to train validation and test sets
        :return:
        """

        item_list = list(pdbfiles_and_chains.items())
        random.Random(seed).shuffle(item_list)

        validation_set = set()
        train_list = []
        banned_pairs = {x[1] for val_set in statistics_map.values() for x in val_set}
        bad_pairs_amount = len(banned_pairs)
        total_pairs = ncr(len(item_list), 2)
        print(f'amount of banned pairs are {bad_pairs_amount} out of a total of {total_pairs}, a total of {bad_pairs_amount/total_pairs} percent')
        inverted_bin_map = {seq[0]: border for border, seq_list in bin_map.items() for seq in seq_list}
        test_list = item_list[:math.floor(self.test_percent*len(item_list))]
        # add to train and validation sets
        for item in item_list[math.ceil(self.test_percent*len(item_list)):]:
            added_to_validation = False
            added_to_test = False
            item_idx, item_name = item
            for test_sample in test_list:
                test_idx, test_name = test_sample
                #Keep chains from same pdb in test set
                if item_name.split('.')[0] == test_name.split('.')[0]:
                    test_list.append(item)
                    added_to_test = True
                    break #same pdb file name
            if added_to_test:
                continue

            # wasn't added to test, check validation
            for test_sample in test_list:
                test_idx, test_name = test_sample
                if (test_idx, item_idx) in banned_pairs or (item_idx, test_idx) in banned_pairs:
                    validation_set.add(item)
                    added_to_validation = True
                    break
            # if the sample isn't too similar to anything in the test set add it to training
            if not added_to_validation:
                train_list.append(item)
        train_set = set(train_list)
        test_set = set(test_list)
        assert not train_set.intersection(validation_set)
        assert not train_set.intersection(test_set)
        assert not validation_set.intersection(test_set)
        test_pdbnames = set([tup[1].split('.')[0] for tup in test_set ])
        validation_pdbnames = set([tup[1].split('.')[0] for tup in validation_set ])
        train_pdbnames = set([tup[1].split('.')[0] for tup in train_set ])
        # assert not train_pdbnames.intersection(validation_pdbnames)
        assert not train_pdbnames.intersection(test_pdbnames)
        assert not validation_pdbnames.intersection(test_pdbnames)
        return set(train_list), validation_set, set(test_list)


    def get_statistics(self, pvalue, cutoff):
        """
        uses similarity matrix to compute statistically significant scores.
        Takes all sequences and puts them into bins according to length.
        We then calculate Pvalue scores for the scores between all samples for every pair of bins.
        Pairs that have a statistically significant high score are those that must be kept in the same train or test set.
        :return: map of pair of bins and the name of the samples that are too similar.
        """

        SEQ_POS = 1
        statistics_map = dict()
        # read sequence map that has the same order as the similarity matrix.
        sequences = self.get_sequences()
        max_sequence_length = max([len(seq[SEQ_POS]) for seq in sequences])
        bins = [(x, x+self.bin_size) for x in range(0, max_sequence_length, self.bin_size)]
        bin_map = dict()
        for border in bins:
            for seq in sequences:
                if border[0] <= len(seq[SEQ_POS]) < border[1]:
                    if border in bin_map:
                        bin_map[border].append(seq)
                    else:
                        bin_map[border] = [seq]
        items = list(bin_map.items())
        # merge bin to equal sizes
        items, bins, bin_map = self.bin_merge(items)
        for bin_tup in combinations_with_replacement(items, 2):
            bin1, bin2 = bin_tup
            border1 = bin1[0]
            border2 = bin2[0]
            scores = self.get_sim_scores_list(bin1, bin2)
            cutoff_set = set([x for x in scores if x[0] > cutoff])
            pvalue_set = set(scores[math.floor((1-pvalue)*len(scores)):])
            # pvalue_set = set()
            statistics_map[(border1, border2)] = cutoff_set | pvalue_set # or operation to merge sets

        return statistics_map, bins, bin_map




    def get_seed_generator(self):
        if len(SEEDS) >= self.k:
            return SEEDS
        else:
            return random.sample(range(1, 10000), 1000)

    def get_redacted_test_list(self, test_list, row_idx, mean_threshold):
        redacted_list = []

        for item in test_list:
            if self.complete_sim_matrix[item[row_idx]].mean() < mean_threshold:
                redacted_list.append(item)

        return redacted_list

    def get_sim_scores_list(self, bin1, bin2):
        IDX = 0
        CONTENT = 1
        BIN_CONTENTS = 1
        sim_scores = []
        for pair in product(bin1[BIN_CONTENTS], bin2[BIN_CONTENTS]):
            idx_seq1, idx_seq2 = pair
            score = self.complete_sim_matrix[idx_seq1[IDX], idx_seq2[IDX]]
            seq_ids = (idx_seq1[IDX], idx_seq2[IDX])
            chain_names = (idx_seq1[CONTENT].description.split('/')[-1], idx_seq2[CONTENT].description.split('/')[-1])
            sim_scores.append((score, seq_ids, chain_names))
        sim_scores = sorted(sim_scores, key=lambda x: x[0])
        return sim_scores

    def get_sequences(self):
        orig_map = np.load(self.RNA_list_path, allow_pickle='TRUE').tolist()
        items = orig_map.items()
        sequences = get_rna_sequences(items, self.fasta_db)
        #TODO remove outliers
        sequences = [seq for seq in sequences if seq.description.split('/')[-1] not in self.banned_sequences]
        return list(enumerate(sequences))

    def get_bins(self, item_arr, inverted_bin_map):

        ret_arr = []
        for item in item_arr:
            ret_arr.append(inverted_bin_map[item[0]])
        return sorted(ret_arr)  # sorted because of the keys

    def drop_similar_samples(self, threshold=0.8):
        """
        looks at similarity matrix and drops both row and column of bad samples
        bad samples are those which have an average similarity above the threshold (0.8 by default)
        """
        dropped_number = 0
        original_size = len(self.sim_matrix)
        for i in range(len(self.sim_matrix)):
            full_matrix = self.sim_matrix.values + self.sim_matrix.values.T
            bad_indexes = np.where(full_matrix.mean(axis=0) > threshold)[0] # for some reason this returns a tuple, get the ndarray at position 0
            if not len(bad_indexes):
                break
            self.banned_sequences.update([self.sim_matrix.columns[bad_indexes[0]]])
            self.sim_matrix = self.sim_matrix.drop(self.sim_matrix.columns[[bad_indexes[0]]], axis=1).drop(self.sim_matrix.index[[bad_indexes[0]]])
            dropped_number += 1
        print(f"after dropping with {self.total_sim_thresh} average similarity, {dropped_number} sequences were dropped out of a total of {original_size}")

    def bin_merge(self, items):
        """
        merge bins to be of roughly equal sizes
        """
        new_bins = []
        cur_list = []
        left_border = -1
        items = sorted(items, key=lambda x: x[0])
        largest_bin = max([len(x[1]) for x in items])

        for i, item in enumerate(items):
            border, samples = item
            if left_border == -1:
                left_border = border[0]
            cur_list += samples
            if len(cur_list) >= largest_bin:
                right_border = border[1]
                new_border = (left_border, right_border)
                new_entry = (new_border, cur_list)
                cur_list = []
                new_bins.append(new_entry)
                left_border = -1

        if cur_list:  # last round wasn't enough, add what we accumulated.
            new_border = (left_border, items[-1][0][1])
            new_bins.append((new_border, cur_list))

        total_bins = [x[0] for x in new_bins]
        bin_map = dict()
        for tup in new_bins:
            bin_map[tup[0]] = tup[1]
        return new_bins, total_bins, bin_map








