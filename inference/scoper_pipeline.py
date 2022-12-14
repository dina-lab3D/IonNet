"""
SCOPER (SAXS-based Conformation Predictor for RNA) tool for finding
conformations of RNA using a SAXS profile.

The program takes as input </path/to/pdb_file> <path/to/saxs_file>
for it to run.
The program requires KGSRNA to be installed locally along with foxs, multifoxs and multifoxs combinations
"""
import os
import subprocess
from inference.inferencePipeline import InferencePipeline
import shutil

class SCOPER:


    def __init__(self, pdb_path: str, saxs_profile_path: str, base_dir: str, inference_type: str
                 , model_path: str, model_config_path: str, saxs_script_path: str, multifoxs_script_path: str,
                 add_hydrogens_script_path: str, kgs_k:int=1):
        """
        constructor for scoper, sets all paths needed to run the pipeline
        :param pdb_path:
        :param saxs_profile_path:
        :param base_dir:
        """
        self.pdb_path = pdb_path
        self.saxs_profile_path = saxs_profile_path
        self.kgs_k = kgs_k
        self.__base_dir = base_dir
        self.__kgsrna_work_dir = os.path.join(self.__base_dir, "KGSRNA")
        self.__saxs_work_dir = os.path.join(self.__base_dir, "saxs_work_dir")
        self.__saxs_script_path = saxs_script_path
        self.__inference_type = inference_type
        self.__model_path = model_path
        self.__model_config_path = model_config_path
        self.__multifoxs_script_path = multifoxs_script_path
        self.__add_hydrogens_script_path = add_hydrogens_script_path

    def run(self):
        """
        logically runs the pipeline, the pipeline works in 6 main stages
        1. validates pdb file
        2. preprocess and runs kgsrna on pdbfile
        3. runs foxs for each sample in kgsrna
        4. sorts saxs scores and runs on the top kgs_k structures
        5. for each remaining structure run inference pipeline
        6. if kgs_k >= 2 we can run multifoxs.
        :return:
        """
        
        top_k_pdbs, kgs_db = KGSRNA(self.__kgsrna_work_dir, self.pdb_path, self.kgs_k, self.__saxs_script_path,
                            self.saxs_profile_path, self.__add_hydrogens_script_path).compute()
        print(top_k_pdbs)
        for pdb_file, _ in top_k_pdbs:  # already sorted
            inference_pipeline = InferencePipeline(os.path.join(os.getcwd(), self.__base_dir), os.path.join(os.getcwd(), os.path.join(os.getcwd(), os.path.join(kgs_db, pdb_file))), self.__inference_type,
                                                   self.__model_path, self.__model_config_path,
                                                   foxs_script=self.__saxs_script_path, multifoxs_script=self.__multifoxs_script_path)
            # inference_pipeline.infer()

        
        

class KGSRNA:
    
    def __init__(self, kgsrna_work_dir: str, pdb_path: str, kgs_k: int, saxs_script_path: str,
                 saxs_profile_path: str, add_hydrogens_script_path: str):
        """
        Initialize kgsrna object
        :param kgsrna_work_dir: where kgsrna samples will be after preprocess
        :param kgsrna_script_path: kgsrna script to run to create samples
        """
        self.kgsrna_work_dir = kgsrna_work_dir
        self.kgsrna_script_path = "scripts/scoper_scripts/Software/Linux64/KGSrna/KGSrna --initial {}.HB --hbondMethod rnaview --hbondFile {}.HB.out -s 1000 -r 20 -c 0.4 --workingDirectory {}/ > ! out "
        self.pdb_path = pdb_path
        self.addhydrogens_script_path = add_hydrogens_script_path  # install locally
        self.rnaview_path = "scripts/scoper_scripts/RNAVIEW/bin/rnaview"
        self.kgs_k = kgs_k
        self.saxs_script_path = saxs_script_path
        self.saxs_profile_path = saxs_profile_path
        self.pdb_workdir = os.path.join(kgsrna_work_dir, os.path.basename(pdb_path))
        self.pdb_workdir_output = os.path.join(self.pdb_workdir, 'output')

    def compute(self):
        """
        runs pipeline stages 2-4
        2. preprocess and runs kgsrna on pdbfile
        3. runs foxs for each sample in kgsrna
        4. sorts saxs scores and returns the top kgs_k structures
        :return: 
        """
        self.preprocess()
        self.get_samples()
        saxs_scores = self.calculate_foxs_scores()
        top_k_pdbs = self.get_top_k(saxs_scores)
        return top_k_pdbs, self.pdb_workdir_output


    def preprocess(self):
        """
        Method that takes the pdb file from pdb_path and does the following before we can run KGSRNA:
        1) Adds hydrogen atoms to pdb file
        2) Run rnaview
        3) Cleans pdb from illegal atom types by deleting certain lines (if they exist)
        4) Creates working directory
        :return:
        """
        print("Adding hydrogens")
        subprocess.run(f"{self.addhydrogens_script_path} {self.pdb_path}", shell=True)
        # set up environment variables for RNAVIEW (must already be installed)
        my_env = os.environ.copy()
        my_env["RNAVIEW"] = f"{os.getcwd()}/scripts/scoper_scripts/RNAVIEW/"
        print("Running rnaview on input pdb")
        subprocess.run(f"{self.rnaview_path} {self.pdb_path}.HB", shell=True, env=my_env)
        self.__kgsrna_clean_pdb()
        if not os.path.isdir(self.kgsrna_work_dir):
            os.mkdir(self.kgsrna_work_dir)
        if not os.path.isdir(self.pdb_workdir):
            os.mkdir(self.pdb_workdir)
            os.mkdir(self.pdb_workdir_output)



    def __kgsrna_clean_pdb(self):
        """
        clean pdb from unwanted lines
        :return:
        """
        ILLEGAL_ATOMS_LIST = ["HO'5", "H21", "H22", "H41", "H42",  "H61", "H62",  "HO'1",
                              "HO'2", "H5'1", "H5'2", "H3T", "H5T"]
        illegal_flag = False
        with open('temp.pdb', 'w') as f:
            with open(f"{self.pdb_path}.HB", 'r') as pdb_file:
                lines = pdb_file.readlines()
                for line in lines:
                    for illegal_atom in ILLEGAL_ATOMS_LIST:
                        if illegal_atom in line:
                            illegal_flag = True
                    if not illegal_flag:
                        f.write(line)
                    illegal_flag = False
        shutil.move('temp.pdb', f"{self.pdb_path}.HB")


    def get_samples(self):
        print("Running KGSRNA with 1000 samples, this may take a few minutes")
        subprocess.run(self.kgsrna_script_path.format(self.pdb_path, self.pdb_path, self.pdb_workdir)
                       , shell=True, stdout=subprocess.DEVNULL)


    def calculate_foxs_scores(self):
        print("Getting foxs scores for 1000 structures")
        saxs_scores = dict()
        for pdb_file in os.listdir(self.pdb_workdir_output):
            if pdb_file.endswith(".pdb"):
                sax_output = subprocess.run(
                    f"{self.saxs_script_path} {os.path.join(self.pdb_workdir_output, pdb_file)} {self.saxs_profile_path}",
                    shell=True, capture_output=True)
                sax_score = float(sax_output.stdout.split()[4])
                saxs_scores[pdb_file] = sax_score
        clean_foxs_files(self.pdb_workdir_output)
        print("Finished scoring")
        return saxs_scores

    def get_top_k(self, saxs_scores: dict):
        sorted_scores = {k: v for k, v in sorted(saxs_scores.items(), key=lambda item: item[1])}
        return list(sorted_scores.items())[:self.kgs_k]


def clean_foxs_files(dirname: str):
    """
    foxs can leave a lot of annoying outputs, this function simply removes them
    given a directory
    :return:
    """
    files = os.listdir(dirname)
    for file in files:
        if file.endswith('dat') or file.endswith('fit'):
            os.remove(os.path.join(dirname, file))


def main():
    pass


if __name__ == '__main__':
    main()
