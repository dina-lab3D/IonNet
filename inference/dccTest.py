"""
This script takes in a model and a list of files / directory of pdb files to run the tests on
Due to the possible large amount of files to process each file will be processed on the cluster.
Takes in the dcc score for each run, averages it and writes the results.
"""

import os
import subprocess
import argparse
import numpy as np
import shutil
from inference.inference_config import config

class DCCTest:

    def __init__(self, model_path: str,  output_path: str, config_path: str = None, dir_path: str = None, list_path: str = None, results_path: str = None,
                 results_dir_path: str = None, overwrite: bool = False, **kwargs):
        self.model_path = model_path
        self.output_path = output_path
        self.config_path = config_path
        self.dir_path = dir_path
        self.list_path = list_path
        self.results_path = os.path.join(os.path.dirname(results_path), os.path.basename(results_path).split('.')[0] +"_"+ os.path.basename(model_path).split('.')[0] + '.txt')
        self.results_dir_path = results_dir_path
        self.results_dir_path = os.path.join(self.results_dir_path, os.path.basename(self.list_path))
        self.overwrite = overwrite # can add to the CMD for overwritting.

    def run_test(self):

        self.dir_path, pdb_test_list = self.__create_test_dir()
        self.__create_results_dir()

        # infer for each file on the cluster.
        script_path = "/cs/usr/punims/Desktop/dina-Lab/punims/MGClassifierV2/scripts/inference/single_dcc_test.sh" if not self.overwrite else "/cs/usr/punims/Desktop/dina-Lab/punims/MGClassifierV2/scripts/inference/single_dcc_test_with_overrwrite.sh"
        interface_dir = "/cs/usr/punims/Desktop/punims-dinaLab/MGClassifier/Interface_grid"
        subprocess.call(f"make -C {interface_dir}", shell=True)  # compiles interface.cc with makefile.
        for file in os.listdir(self.dir_path):
            if file in pdb_test_list and file not in config['banned']:
                slurm_output_path = os.path.join(self.output_path, f'{os.path.basename(file)}_slurm.out')
                output_result_path = os.path.join(self.results_dir_path, os.path.basename(file)) + '_result.txt'
                # each run writes to the output file
                CMD = f"sbatch --killable --output={slurm_output_path} {script_path} {os.path.join(self.dir_path, file)} {self.model_path} {self.config_path} {output_result_path} {self.output_path}"
                if self.overwrite:
                    CMD = CMD + f" {self.overwrite}"
                subprocess.run(CMD, shell=True, executable='/bin/csh')

    def calculate_result(self):
        if os.path.isfile(self.results_path):
            os.remove(self.results_path)
        lines = []
        for file in os.listdir(self.results_dir_path):
            if '.pdb' in file:
                with open(os.path.join(self.results_dir_path,file), 'r') as f:
                    file_lines = f.readlines()
                    for line in file_lines:
                        lines.append(line)
        with open(self.results_path, 'w'):
            # rewrite output file
            pass
        if self.results_path:
            with open(self.results_path, 'w') as f:
                f.writelines(lines)
                results = [float(line.split()[-2]) for line in lines]
                total_atoms = [int(line.split()[-1]) for line in lines]
                weighted_mean = np.average(results, weights=total_atoms)
                print(f'in total the dcc weighted mean is {weighted_mean} out of a total of {sum(total_atoms)}')


    def __create_test_dir(self):
        """
        Assuming the file is of the kfp format, find line that begins test set and to each pdb file from test set
        add the base dir of the database, then create a new directory in the inference folder and copy all files
        from the database to the new folder
        @return:
        """
        with open(self.list_path) as f:
            lines = f.readlines()
            for x, line in enumerate(lines):
                if line.strip() == "Test Examples:":
                    break
            test_pdb_files = []
            for line in lines[x+1:]:
                pdb_name = line.split()[0]
                if pdb_name.endswith('.pdb'):
                    test_pdb_files.append(pdb_name)
                else:
                    og_files = [os.path.join("/cs/usr/punims/Desktop/dina-Lab/punims/Databases/Database_1_MG_RNA", file) for file in test_pdb_files]
                    dest = "/cs/usr/punims/Desktop/dina-Lab/punims/MGClassifierV2/inference/pdbfiles"
                    for file in og_files:
                        shutil.copy(file, dest)
                    return dest, test_pdb_files

    def __create_results_dir(self):
        """
        Results should be in different folders depending on what test set was used.
        This function simply creates a results directory using the base results path and adding the name of the
        test set (partition file used)
        """

        if not os.path.isdir(self.results_dir_path):
            os.mkdir(self.results_dir_path)





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mp', '--model-path', type=str, required=True)
    parser.add_argument('-bd', '--base-dir', type=str, required=True)
    parser.add_argument('-kfp', '--kfold-path', type=str, required=True)
    parser.add_argument('-cp', '--config-path', type=str, required=True)
    parser.add_argument('-op', '--output-path', type=str, required=True)
    parser.add_argument('-dp', '--dir-path', type=str, required=False, default=None)
    parser.add_argument('-lp', '--list-path', type=str, required=False, default=None)
    parser.add_argument('-rp', '--results-path', type=str, required=False, default=None)
    parser.add_argument('-rdp', '--results-dir-path', type=str, required=False, default=None)
    parser.add_argument('-rb', '--result-bool', type=bool, default=False)
    parser.add_argument('-ov', '--overwrite', type=bool, default=False)

    args = parser.parse_args()
    tester = DCCTest(**vars(args))

    if args.result_bool:
        tester.calculate_result()
    else:
        tester.run_test()

if __name__ == '__main__':
    main()