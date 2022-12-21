"""
Script to run multi_foxs program
takes in as input <dir1> <dir2> <dir3> ... output_name, sax_profile_path
all directories containing PDB files.
preprocess: The script first adjusts the selected probes and turns them into Mg ions
it then combines the selected probes and the RNA to a new pdb file and writes the path to a file.
calculate_saxs_profile: takes the output file from the preprocess stage and runs foxs on each file creating a saxs profile

run_multifoxs: After all combined pdbs are created and all saxs profiles are calculated  multifoxs is called on the
sax profiles
"""
import math
import os
import subprocess
from inference.inference_utils import fix_probe_lines, combine_files
import sys
from inference.inference_config import config
from multiprocessing import Pool


class MultiFoxs:
    OUTPUT_FILE_NAME = "out.txt"
    SAXS_OUTPUT_FILE_NAMES = "sax_out.txt"
    FOXS_FLAGS = " -p --max_c1 1.00 --min_c1 1.00 --min_c2 1.00 --max_c2 1.00"
    MULTIFOXS_FLAGS = " -t 0 --max_c1 1.00 --min_c1 1.00 --min_c2 1.00 --max_c2 1.00"

    def __init__(self, dirs: [str], output: str, profile: str = None, overwrite: bool = False, foxs_script_path: str = None,
                 multifoxs_script_path: str = None):
        self.dirs = dirs
        self.output = os.path.join(os.getcwd(), output)
        self.profile = profile
        self.result_path = os.path.join(self.output, "result.txt")
        self.overwrite = overwrite
        self.foxs_script_path = foxs_script_path + self.FOXS_FLAGS
        self.multifoxs_script_path = multifoxs_script_path + self.MULTIFOXS_FLAGS

        if profile is None:
            self.profile = config['sax_path']

        if not os.path.isdir(output):
            os.mkdir(self.output)

    def run(self):
        if not os.path.isfile(self.result_path) or self.overwrite:
            self.preprocess()
            self.calculate_saxs_profiles()
            self.run_multifoxs()
            self.save_results(self.output, 'has MG')
        else:
            print(f'Results already exist for {self.output}')
        return self.get_results()


    def preprocess(self):
        with open(self.OUTPUT_FILE_NAME, 'w') as f:
            for my_dir in self.dirs:
                for conformation_dir in os.listdir(my_dir):
                    if os.path.isdir(os.path.join(my_dir, conformation_dir)):
                        abs_conformation_dir = os.path.join(my_dir, conformation_dir)
                        print(f'working on {abs_conformation_dir}')
                        basename_dir = os.path.basename(conformation_dir)
                        rna_path = os.path.join(abs_conformation_dir, f"{basename_dir}_rna_.pdb")
                        try:
                            mg_path = self.__convert_probes_to_mg(abs_conformation_dir)
                        except FileNotFoundError:
                            print(f"dir: {conformation_dir} has no predictions, skipping directory")
                            continue
                        combined_path = combine_files(rna_path, mg_path, f"combined_{conformation_dir}_rna_mg.pdb",
                                                      abs_conformation_dir)
                        f.write(combined_path + '\n')


    def calculate_saxs_profiles(self):
        # First run with MG
        with open(self.OUTPUT_FILE_NAME, 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        with Pool(8) as p:
            p.map(self.calculate_saxs_profiles_helper, lines)

    def calculate_saxs_profiles_helper(self, line):
        subprocess.run(f"{self.foxs_script_path} {line}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def run_multifoxs(self):
        # First run with MG
        with open(self.OUTPUT_FILE_NAME, 'r') as f:
            lines = f.readlines()
        lines = [line.strip() + '.dat\n' for line in lines]
        with open(self.SAXS_OUTPUT_FILE_NAMES, 'w') as f:
            f.writelines(lines)
        print("Running MultiFoXs script")
        subprocess.run(f"{self.multifoxs_script_path} {self.profile} {os.path.abspath(self.SAXS_OUTPUT_FILE_NAMES)}",
                       shell=True, cwd=self.output, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


    def __convert_probes_to_mg(self, conformation_dir):
        basename_dir = os.path.basename(conformation_dir)
        probe_path = os.path.join(conformation_dir, f"new_probe_{basename_dir}_probes.pdb")
        fixed_mg_path = os.path.join(conformation_dir, f"fixed_mg.pdb")
        with open(probe_path, 'r') as f:
            lines = f.readlines()
        lines = fix_probe_lines(lines)
        with open(fixed_mg_path, 'w') as f:
            f.writelines(lines)
        return fixed_mg_path

    def save_results(self, output_dir, dir_type):

        best_result = math.inf
        for file in os.listdir(output_dir):
            if 'ensemble' in file:
                with open(os.path.join(output_dir, file), 'r') as f:
                    first_line = f.readline()
                    score = float(first_line.split('|')[1])
                    if score < best_result:
                        best_result = score
        with open(self.result_path, 'w') as f:
            f.write(str(best_result) + " " + dir_type + '\n')
        return best_result

    def get_results(self):
        if os.path.isfile(self.result_path):
            with open(self.result_path, 'r') as f:
                score_with_mg = float(f.readline().split(' ')[0])
                return score_with_mg


def main():
    dirs = sys.argv[1:-2]
    output = sys.argv[-2]
    sax_profile = sys.argv[-1]
    foxs = MultiFoxs(dirs, output, sax_profile)
    foxs.run()


if __name__ == '__main__':
    main()
