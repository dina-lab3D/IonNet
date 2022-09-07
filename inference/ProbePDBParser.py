"""
My own parser class for PDBs that contain probes (they don't follow the rules of regular PDBs)
"""

import numpy as np

class ProbePDBParser:

    def get_structure(self, pdb_file):
        with open(pdb_file, 'r') as f:
            lines = f.readlines()
            if len(lines) == 0:
                raise ValueError("Empty File")
            self.structure = ProbePDBParser.Structure(lines)
            return self.structure

    class Structure:
        def __init__(self, lines: [str]):
            lines = lines
            self.atoms = []
            for line in lines:
                self.atoms.append(ProbePDBParser.Atom(line))

        def get_atoms(self):
            return self.atoms

    class Atom:

        def __init__(self, line: str):
            X_BEGIN = 31
            X_END = 38
            Y_BEGIN = 39
            Y_END = 46
            Z_BEGIN = 47
            Z_END = 54
            self.coordinates = np.array([float(line[X_BEGIN: X_END]), float(line[Y_BEGIN: Y_END]), float(line[Z_BEGIN: Z_END])])

        def get_coord(self):
            return self.coordinates

