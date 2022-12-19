from inference.ProbePDBParser import ProbePDBParser
import math
from scipy.spatial import distance
import os

"""
Useful utility functions most inference classes can use. 
"""

def extract_atom_coordinates(file: str):
    """
    func takes a pdb file, opens it with Biopython and extracts the coordinates for each atom in the file
    """
    parser = ProbePDBParser()
    structure = parser.get_structure(file)
    atoms = structure.get_atoms()
    coordinates = [atom.get_coord() for atom in atoms]
    return coordinates

def find_shortest_dist(target, sources):
    best_dist = math.inf
    for source in sources:
        cur_dist = distance.euclidean(target, source)
        if cur_dist < best_dist:
            best_dist = cur_dist
    return best_dist


def dist_check(coords, thresh):
    return lambda other_coords: distance.euclidean(coords, other_coords) < thresh

def combine_files(file1: str, file2: str, file_combined: str, process_odir):
    """

    :param file1: file path for the first file
    :param file2: file path for the second file
    :param file_combined: file path for a new file in which the contents of 1 and 2 should be concatenated
    opens files concatenates them into the third and then closes them all.
    :return: combined file path
    """

    with open(file1, "r") as fp:
        data1 = fp.read()

    with open(file2, "r") as fp:
        data2 = fp.read()

    data1 += '\n'
    data1 += data2

    with open(os.path.join(process_odir, file_combined), "w") as fp:
        fp.write(data1)
        return os.path.realpath(fp.name)

def fix_probe_lines(lines: [str]):
    """
    takes PDB lines as input, changes lines inplace:
    ATOM changes to HETATM, PB changes to MG, UNK changes to MG and the last line changes to MG
    :param lines: the pdb lines
    :return: the altered lines
    """

    new_lines = []
    for line in lines:
        line = "HETATM" + line[6:]
        line = line[:12] + "MG " + line[15:]
        line = line[:17] + " MG" + line[20:]
        line = line.rstrip()
        line += 10*" " + "MG\n"
        new_lines.append(line)

    return new_lines

def find_chi_score(stdout):
    """
    When running foxs with the new IMP version it can have warnings.
    This method simply finds the line where Chi^2 exists and returns the score
    :return:
    """
    search_token = "Chi^2"
    lines = str(stdout).split("\n")
    for line in lines:
        if search_token in line:
            return float(line.split()[4])

