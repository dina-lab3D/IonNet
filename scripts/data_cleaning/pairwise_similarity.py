import numpy as np

import Bio.SeqIO.FastaIO
import os
import subprocess
import sys

from Bio import pairwise2
from tqdm import tqdm
from config import *
import re


ARGC = 4
LIST = 1
FASTA_DB = 2
OUTPUT = 3

ARGC_ERR = "inputs should be: path to RNA list, fasta database path, output path"


def get_rna_sequences(items: list, fasta_db: str):
    """
    Given a list of dictionary items (tuple of key, values)
    take key to open fasta file in the fasta_db directory. then using the chains extracts the correct sequence
    and adds it to a list which will be returned at the end
    :param items:
    :return:
    """

    CHAIN_IDX = 3

    ret_arr = []
    for file, chains in items:
        file = file.split('.')[0]
        fasta_path = os.path.join(fasta_db, file+".fasta")
        record = list(Bio.SeqIO.parse(fasta_path, "fasta"))
        for seq in record:
            # We made sure when making the items that chains are part of an RNA
            # But the same can't be said for the fasta files, so we need to filter out the RNA chains
            chain_name = seq.description.split(" ")[CHAIN_IDX]
            if chain_name in chains:
                chains.remove(chain_name)  # for some reason some pdb files have the same chain name twice!
                ret_arr.append(seq)

    return ret_arr


def get_file_numbers(path):
    """
    returns list of numbers that are contained in filenames in the path
    :param path:
    :return:
    """

    file_numbers = []
    for filename in os.listdir(path):
        numbers = re.findall(r"\d+", filename)
        file_numbers += numbers
    return file_numbers


def calc_pairwise_similarity_for_RNA_DB(RNA_list_path: str, fasta_db:str,  output_path: str):
    """
    Given a path to a dictionary, keys being PDB filenames and values being which chains in the PDB file are
    part of RNA chains, calculates pairwise sequence identity for every two RNA chains and saves the scores.
    :param RNA_list_path:
    :param output_path:
    :return:
    """

    orig_map = np.load(RNA_list_path, allow_pickle='TRUE').tolist()
    items = orig_map.items()
    sequences = get_rna_sequences(items, fasta_db)
    file_numbers = get_file_numbers(PAIRWISE_ROWS_DIR)
    # pairwise calculations for the sequences:
    # can calculate upper triangle of the distance matrix to save time n(n+1)/2 instead of n*n
    for i, seq1 in tqdm(enumerate(sequences)):
        if str(i) not in file_numbers:
            print(f"Sending row {i}")
            subprocess.run(f"sbatch --killable --output=R-%x.row{i}.out pairwise_bash.sh {i} {fasta_db} {RNA_list_path}", shell=True, executable='/bin/csh')
        else:
            print(f"row {i} already contained in output directory")


def main():
    """
    calculates pairwise similarity, arguments that should be passed:
    RNA list path (output from data_cleaning.create_RNA_chain_list)
    database with fasta files (output of transform_pdb2fasta)
    a path to a file that will be used to save the pairwise calculation.
    """
    args = sys.argv
    if len(args) != ARGC:
        print(ARGC_ERR, file=sys.stderr)
        return
    calc_pairwise_similarity_for_RNA_DB(sys.argv[LIST], sys.argv[FASTA_DB], sys.argv[OUTPUT])


if __name__ == '__main__':
    main()
