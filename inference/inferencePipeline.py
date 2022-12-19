"""
This file contains the inferencePipeline class that is responsible for taking a PDB file and a graph prediction model.
The class is made of two main parts. One that preprocesses the PDB file and the second does the predicting.
"""

import random
import subprocess
from typing import Tuple
import torch
import shutil
from models.training import GraphGNN
from config import base_config
import numpy as np
from preprocessing.dataset import get_dataset
import matplotlib.pyplot as plt
from inference.inference_config import config
from inference.inference_utils import *
import matplotlib.image as mpimg
import matplotlib.ticker as mticker


class InferencePipeline:

    def __init__(self, odir: str, fpath: str, inference_type: str, model_path: str, model_config_path: str = None, pymol=False, overwrite=False, foxs_script=None, multifoxs_script=None):
        self.processor = PDBProcessor(odir, fpath)
        self.predictor = Predictor(model_path, model_config_path)
        self.presenter = InferencePresenter(inference_type, pymol, fpath)
        self.overwrite = overwrite
        config['foxs_script'] = foxs_script
        config['multifoxs_script'] = multifoxs_script

    def infer(self):
        features_dir, probe_path, rna_path, mg_path = self.processor.process()
        self.predictions_path = os.path.join(os.path.dirname(probe_path), f'{os.path.basename(self.predictor.model_path).split(".")[0]}_predictions.npy')
        dataset = get_dataset('inference_dataset', features_dir)
        predictions = self.__load_predictions(dataset)
        predicted_probe_path = self.presenter.show(predictions, rna_path, probe_path, mg_path, dataset)
        return {'predictions': predictions,
                'predicted_probe_path': predicted_probe_path,
                'rna_path': rna_path,
                'probe_path': probe_path,
                'mg_path': mg_path}

    def test(self, predicted_probe_path, mg_path, dcc_thresh=4, **kwargs):
        test_metrics = TestMetrics(predicted_probe_path, mg_path, dcc_thresh, **kwargs)
        test_metrics.run_tests()

    def cleanup(self):
        """
        deletes features directory, delegated to the processor that created the directory.
        @return:
        """
        self.processor.cleanup()

    def __load_predictions(self, dataset):
        if not os.path.isfile(self.predictions_path) or self.overwrite:
            if self.overwrite:
                print("predictions made but overwriting them")
            else:
                print('predictions not made before, calculating')
            predictions = self.predictor.predict(dataset)
            np.save(self.predictions_path, predictions)
        else:
            print('predictions saved before, loading')
            predictions = np.load(self.predictions_path, allow_pickle=True)
        return predictions

class PDBProcessor:

    def __init__(self, odir: str, fpath: str):
        self.odir = odir
        self.fpath = fpath
        self.process_odir = os.path.join(odir, os.path.basename(self.fpath).split('.')[0])
        self.process_odir_combined = os.path.join(self.process_odir, "combined_dir")
        self.add_mg = False

    def process(self):
        self.__create_output_folder(self.odir)
        if os.path.isdir(self.process_odir):
            print(f"Already created raw files for {self.fpath}")
            file_name, extension = os.path.splitext(os.path.basename(self.fpath))
            feature_dir = os.path.join(self.process_odir, 'features')
            probe_path = os.path.join(self.odir, file_name, "{}_probes{}".format(file_name, extension))
            rna_path = os.path.join(self.odir, file_name, f'{file_name}_rna_{extension}')
            mg_path = os.path.join(self.odir, file_name, f'{file_name}_mg_{extension}')
            return feature_dir, probe_path, rna_path, mg_path
        self.__create_output_folder(self.process_odir)
        self.__create_output_folder(self.process_odir_combined)
        probe_path, rna_path, mg_path, file_name, extension = self.__create_surface_RNA_mg(self.fpath, self.process_odir)
        combined_name = "combined_" + file_name + extension
        combined_path = combine_files(self.fpath, probe_path, combined_name, self.process_odir)
        shutil.move(combined_path, self.process_odir_combined)
        features_dir = self.__create_features(self.process_odir_combined, self.process_odir, self.add_mg)
        return features_dir, probe_path, rna_path, mg_path

    def cleanup(self):
        """
        delete features directory
        @return:
        """
        features_dir = os.path.join(self.process_odir, 'features')
        shutil.rmtree(features_dir)


    def __create_output_folder(self, odir: str):
        """
        creates a new directory named odir
        :param odir:
        """

        try:
            os.mkdir(odir)
        except FileExistsError as e:
            print("Directory already exists")
            return
        except OSError as e:
            print("Failed to create %s directory" %odir)
            raise e
        else:
            print("Successfully created the directory %s " % odir)


    def __create_surface_RNA_mg(self, fpath: str, odir: str) -> Tuple[str, str, str, str, str]:
        """
        Creates three pdb files using C++ compiled code.
        A connolly surface pdb file for the probes, a file with just the RNA backbone and a file with the
        MG ions.
        located at /cs/usr/punims/punims/MGClassifierV2/inference/pdb_processing/generatePDB
        :param odir: output directory absolute path
        :param fpath: path of PDB which we want to prepare
        :return: path of the PDB files which we created, original filename and extension
        """

        WORK_DIR = os.getcwd()+"/inference/pdb_processing/"
        SURFACE_PROG = WORK_DIR+"/generatePDB"
        file_name, extension = os.path.splitext(os.path.basename(fpath))
        probe_path = odir + "/{}_probes{}".format(file_name, extension)
        rna_path = odir + f'/{file_name}_rna_{extension}'
        mg_path = odir + f'/{file_name}_mg_{extension}'
        # subprocess run waits until process is finished so we can be pretty sure the files are made if successful.
        subprocess.run(SURFACE_PROG + f' {fpath} {probe_path} {rna_path} {mg_path}', shell=True, cwd=WORK_DIR, stdout=subprocess.DEVNULL)
        return probe_path, rna_path, mg_path, file_name, extension

    def __create_features(self, combined_path: str, output_dir: str, add_mg: bool = False):
        """
        creates a new folder with output_dir (if it already exists nothing new is made)
        within output_dir we create a feature folder which is where all features will be placed.
        we then move our combined file from its current path into the output_dir
        from there we create all the feature vectors from the probe atoms
        :param combined_path: combined pdb file of an RNA and the surface probes
        :param output_dir: the output directory folder path, within it we create a new feature folder
        :return: return path of the combined path
        """

        import shutil

        # define the name of the directory to be created
        features_dir = os.path.join(output_dir, "features")
        feature_raw = os.path.join(features_dir, "raw")  # calling it raw is important for pyg dataset
        self.__create_output_folder(features_dir)
        self.__create_output_folder(feature_raw)
        self.__create_output_folder(os.path.join(features_dir, 'processed'))
        interface_dir = os.getcwd()+"/Interface_grid/"
        file_name, extension = os.path.splitext(os.path.basename(combined_path))
        new_combined_path = features_dir + '/' + file_name + extension
        shutil.move(combined_path, new_combined_path)
        INTERFACE2GRID = interface_dir+"/interface2grid -i " + new_combined_path + " -o " + feature_raw + " --selector PB --voxel-size 1.0 -x 32 -y 32 -z 32 -r 8.0 --graph_representation --probe"
        print(f"Running command {INTERFACE2GRID}")
        subprocess.run(INTERFACE2GRID, shell=True, cwd=interface_dir)
        if add_mg:
            print("adding mg as well")
            mg_pdb = os.path.join(feature_raw, "mg_pdb")
            mg_out = os.path.join(feature_raw, "mg_raw")
            self.__create_output_folder(mg_pdb)
            self.__create_output_folder(mg_out)
            shutil.copy(self.fpath, os.path.join(mg_pdb, "original_"+os.path.basename(self.fpath)))
            INTERFACE2GRID_MG = "interface2grid -i " + mg_pdb + " -o " + mg_out + " --selector MG --voxel-size 1.0 -x 32 -y 32 -z 32 -r 8.0 --graph_representation"
            subprocess.run(INTERFACE2GRID_MG, shell=True, cwd=interface_dir)
            for file in os.listdir(mg_out):
                shutil.move(os.path.join(mg_out, file), feature_raw)
            os.rmdir(mg_out)
        print("Finished creating raw features")
        return features_dir


class Predictor:

    def __init__(self, model_path: str, model_config_path: str):
        self.model_path = model_path
        self.model = GraphGNN(base_config)  # need to pass a basic dictionary, all values will be overridden in the load
        print(f"loading model {self.model_path}")
        self.model.load(self.model_path, 'eval', model_config_path)

    def predict(self, dataset):
        predictions = self.model.test(dataset, base_config['test_dict']['thresh'], inference=True)
        return predictions


class InferencePresenter:
    """
    This class is responsible for presenting the results of the prediction.
    To do this the class should rewrite the results to a PDB using one of the inference types:
    top: colors the top X% of predictions
    sax: adds the top predictions one at a time, each time writing a new PDB file and running the SAX experiment.
    as long as the score keeps dropping (or doesn't go up beyond some threshold) we keep adding ions.

    At the end of the process the PDB file should open in pymol for viewing.
    """
    def __init__(self, inference_type: str, pymol: bool, fpath: str):
        self.inference_type = inference_type
        self.inference_types = ['top', 'sax', 'cluster', 'random']
        self.fpath = fpath
        if self.inference_type not in self.inference_types:
            raise Exception(f"{self.inference_type} not a valid inference type, try again")
        self.func_map = {'top': Top,
                         'sax': SAX,
                         'cluster': self.__cluster_present,
                         'random': self.__random_present}
        self.pymol = pymol

    def show(self, predictions, rna_path: str, probe_path: str, mg_path: str, dataset):
        predictions = self.__sort_according_to_lines(predictions, dataset)
        predictions = torch.FloatTensor(predictions).detach().cpu().numpy()
        kwargs = {'cutoff': base_config['dcc_cutoff'],
                  'sax_path': config['sax_path'],
                  'safe_dist': config['safe_dist'],
                  'probe_path': probe_path,
                  'mg_path': mg_path,
                  'rna_path': rna_path,
                  'dataset': dataset,
                  'predictions': predictions,
                  'odir': os.path.dirname(probe_path),
                  'fpath': self.fpath,
                  'combined_sax': config['combined_sax']}
        new_labels = self.func_map[self.inference_type](**kwargs)()
        return self.__create_and_present_pdb(new_labels, rna_path, probe_path, mg_path)

    def __top_present(self, predictions, cutoff=0.1, **kwargs):
        """
        read surface path file, creates array of coordinates.
        takes cutoff of predictions (None to All)
        :param prediction_list: list of prediction labels {0,1}
        :param confidence_list: model's confidence in the prediction [0,1]
        :param probe_path: PDB file path
        """

        prediction_label = predictions > config['positive_thresh']
        new_predictions = [0]*len(prediction_label)
        confidence_list_positive = [x[0] for x in zip(predictions, prediction_label) if x[1] == 1]
        index_list = list(range(len(prediction_label)))
        index_list_positive = [x[0] for x in zip(index_list, prediction_label) if x[1] == 1]
        top = sorted([x for x in zip(index_list_positive, confidence_list_positive)], key=lambda x: x[1])
        top = top[math.floor(-len(top)*cutoff):]
        for i, _ in top:
            new_predictions[i] = 1
        return new_predictions

    def __random_present(self, predictions, cutoff=0.1, **kwargs):
        """
        randomly returns labels, cutoff being the proportion of positive labels for it to guess
        @param kwargs:
        @return:
        """

        total_labels = len(predictions)
        label_list = [0] * math.floor((1-cutoff) * total_labels) + [1] * math.ceil(cutoff*total_labels)
        random.shuffle(label_list)
        return label_list

    def __cluster_present(self, predictions, **kwargs):
        pass

    def __color_pdb(self, rna_path: str, probe_path: str, mg_path: str):
        pass

    def __create_and_present_pdb(self, new_labels, rna_path: str, probe_path: str, mg_path: str):
        root, base = os.path.split(probe_path)
        new_probe_output_path = os.path.join(root, "new_probe_"+base)
        self.__create_new_probes_pdb(probe_path, new_probe_output_path, new_labels)
        if self.pymol:
            print("running " + "pymol " + "{} {}".format(new_probe_output_path, rna_path))

            subprocess.run(f"pymol " + f"{new_probe_output_path} {rna_path} {mg_path}", shell=True,
                           executable="/bin/csh", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return new_probe_output_path

    def __create_new_probes_pdb(self, probe_file_path: str, output_name: str, label_array):
        """
        create a new pdb file using the old probe file, given an array of labels, taking only positive probes.
        :param cluster: Whether or not to cluster positive labels. if True only add representatives of each cluster.
        :param label_array: array of labels
        :param probe_file_path: string of the probe file path
        :param output_name: name of the new output file.
        """

        with open(probe_file_path) as probe_fp:
            lines = probe_fp.readlines()


            with open(output_name, "w") as colored_fp:
                for line, label in zip(lines, label_array):
                    if label == 1:
                        colored_fp.write(line)

    def __sort_according_to_lines(self, predictions, dataset):
        """
        The dataset opens up not according to the order of the lines of the probes, this method sorts the predictions
        to match their number line using the data in the dataset.
        @param predictions:
        @param dataset:
        @return:
        """
        true_indexes = [int(data.name.split("_")[-1]) for data in dataset]
        new_predictions = [0] * len(true_indexes)
        for i, val in zip(true_indexes, predictions):
            new_predictions[i] = val
        return new_predictions


class Top:

    IDX = 0
    COORDS = 1
    PRED = 2

    """
    Class that presents the top predictions of the model. 
    The class does iterative clustering so that only the probe with the highest confidence in a certain radius is 
    chosen (1.5 angstrom according to MetalIonRNA is a good cutoff)
    """

    def __init__(self, predictions: np.ndarray, sax_path, odir, probe_path, safe_dist, **kwargs):

        self.odir = odir
        self.sax_path = sax_path
        self.predictions = predictions
        self.probe_path = probe_path
        self.safe_dist = safe_dist

    def __call__(self, *args, **kwargs):
        """
        access method to the sax pipeline
        @return:
        """
        candidates = self.__get_candidates()
        predicted_indexes = [candidate[self.IDX] for candidate in candidates]
        new_labels = [0] * len(self.predictions)
        for i in predicted_indexes:
            new_labels[i] = 1
        return new_labels

    def __get_candidates(self):
        candidates = []
        labels = self.predictions > config['positive_thresh']
        confidence_list_positive = [x[0] for x in zip(self.predictions, labels) if x[1] == 1]
        index_list = list(range(len(labels)))
        coordinates = extract_atom_coordinates(self.probe_path)
        index_and_coord_list_positive = [x[0] for x in zip(zip(index_list, coordinates), labels) if x[1] == 1]
        index_positive, coords_positive = zip(*index_and_coord_list_positive)
        top = sorted([x for x in zip(index_positive, coords_positive, confidence_list_positive)], key=lambda x: x[2])
        top.reverse()
        for item1 in top:
            dist_func = dist_check(item1[self.COORDS], self.safe_dist)
            adversary_indexes = []
            for i, item2 in enumerate(candidates):
                if dist_func(item2[self.COORDS]):
                    adversary_indexes.append(i)
                    break
            # sorted from top to bottom, we know the confidence is higher for all existing candidates
            if adversary_indexes:
                continue
            else:
                candidates.append(item1)
        return candidates

class SAX:
    """
    Class responsible for calculating what atoms should be apart of the sax experiment.
    Also runs the sax experiment itself
    """

    IDX = 0
    COORDS = 1
    PRED = 2
    FIXED_FILE_NAME = "fixed_pdb.pdb"
    COMBINED_FILE_NAME = 'combined_rna_mg.pdb'

    def __init__(self, predictions: np.ndarray, sax_path, odir, probe_path, safe_dist, mg_path, rna_path, fpath, combined_sax, **kwargs):

        self.odir = odir
        self.sax_path = sax_path
        self.predictions = predictions
        self.probe_path = probe_path
        self.safe_dist = safe_dist
        self.mg_path = mg_path
        self.rna_path = rna_path
        self.fpath = fpath
        self.combined_sax = combined_sax
        self.SAX_SCRIPT = config['foxs_script'] + " --min_c1 1.02 --max_c1 1.02 --min_c2 1.00 --max_c2 1.00 {} {}"
        self.SAX_SCRIPT_COMBINED = config['multifoxs_script']
        self.SAX_SCRIPT_VANILLA = config['foxs_script'] + " {} {}"

    def __call__(self, *args, **kwargs):
        """
        access method to the sax pipeline
        @return:
        """
        labels = Top(**self.__dict__)()
        if self.combined_sax:
            probes_chosen_by_sax = self.__run_combined_sax_script(labels)
        else:
            probes_chosen_by_sax = self.__run_incremental_sax(labels)
        return probes_chosen_by_sax


    def __run_incremental_sax(self, labels):
        """
        helper function for the sax prediction list, which adds MG atoms incrementally according to the SAX script score
        :return:
        """

        # create files needed to run that is a combination of the RNA file with MG


        with open(self.probe_path) as probe_fp:
            lines = probe_fp.readlines()
            #fix surface path text
            self.__create_required_files()
            lines = fix_probe_lines(lines)
            score = math.inf
            scores = []
            sax_labels = []
            combine_files(self.rna_path, self.mg_path, os.path.join(self.odir, "rna_with_original_mg.pdb"), self.odir)
            sax_output = subprocess.run(self.SAX_SCRIPT.format(os.path.join(self.odir, "rna_with_original_mg.pdb"), self.sax_path), shell=True, capture_output=True)
            sax_score = find_chi_score(sax_output.stdout)
            print(f"the beginning score is {sax_score}")
            for i, label in enumerate(labels):
                with open(os.path.join(self.odir, self.FIXED_FILE_NAME), "a") as fixed_file:
                    if label:  # 0 is the index list part of the zip
                        fixed_file.write(lines[i])
                    else:
                        sax_labels.append(0)
                        continue

                combine_files(self.rna_path, os.path.join(self.odir, "fixed_pdb.pdb"), f"RNA_with_{i}_MG.pdb", self.odir)

                #score sax
                sax_output = subprocess.run(self.SAX_SCRIPT.format(os.path.join(self.odir, f"RNA_with_{i}_MG.pdb"), self.sax_path), shell=True, capture_output=True)
                sax_score = find_chi_score(sax_output.stdout)
                print(f"the cur score is {sax_score}")
                epsilon = 0.05

                #if score is up to some epsilon or passes real score, break
                if sax_score <= score or sax_score - score < epsilon:
                    score = sax_score
                    scores.append(score)
                    sax_labels.append(1)
                else:
                    sax_labels.append(0)
                    # remove last line
                    with open(os.path.join(self.odir, self.FIXED_FILE_NAME), "r+") as fixed_file:
                        fixed_file.seek(0, os.SEEK_END)
                        pos = fixed_file.tell() - 1
                        while pos > 0 and fixed_file.read(1) != "\n":
                            pos -= 1
                            fixed_file.seek(pos, os.SEEK_SET)
                        if pos > 0:
                            fixed_file.seek(pos, os.SEEK_SET)
                            fixed_file.truncate()
                            fixed_file.write('\n')

        # run saxs but this time with the c1 and c2 that can be fit
        final_product_name = "RNA_final.pdb"
        combine_files(self.rna_path, os.path.join(self.odir, "fixed_pdb.pdb"), final_product_name, self.odir)
        sax_output = subprocess.run(self.SAX_SCRIPT_VANILLA.format(os.path.join(self.odir, final_product_name), self.sax_path), shell=True, capture_output=True)
        sax_score = find_chi_score(sax_output.stdout)
        print(f"the final score with fitting c1 and c2 is {sax_score}")
        return sax_labels

    def __create_required_files(self):
        """
        Creates the combined RNA and MG files
        restarts iterative file.
        @return:
        """
        fp = open(os.path.join(self.odir, self.FIXED_FILE_NAME), "w") #overwrite old file
        fp.close()

    def __create_combined_required_folder_and_files(self, lines, labels):
        """

        @param lines: lines from the mg probe file that need to be changed into mg atom lines.
        @return: mg ion folder path, saxs work directory
        """
        lines = fix_probe_lines(lines)
        mg_folder_path = os.path.join(os.path.dirname(self.mg_path), os.path.basename(self.mg_path).split('.')[0])
        sax_work_dir = os.path.join(self.odir, 'sax_work_dir')
        sax_work_dir_current_dat = os.path.join(sax_work_dir, os.path.basename(config["sax_path"]))
        if os.path.exists(mg_folder_path):
            print("mg dir already exists, removing it")
            shutil.rmtree(mg_folder_path)
        os.mkdir(mg_folder_path)
        if os.path.exists(sax_work_dir):
            print("sax work dir already exists, removing it")
            shutil.rmtree(sax_work_dir)
        os.mkdir(sax_work_dir)
        os.mkdir(sax_work_dir_current_dat)
        for i, label in enumerate(labels):
            if label:
                cur_mg_path = os.path.join(mg_folder_path, f'mg_{i}.pdb')
                with open(cur_mg_path, 'w') as f:
                    f.write(lines[i])
        return mg_folder_path, sax_work_dir_current_dat



    def __run_combined_sax_script(self, labels):
        """
        Runs combined sax script.
        The combined sax script takes in the dat file, rna file and each mg in its own pdb file
        it then creates the best combination it can find using a hierarchical heuristic that checks the 'k' best
        pairs then the 'k' best quadruples and so on until a combination does not yield any lower saxs score.
        @param labels: labels for the probes saying which one is positive and negative according to the model.
        @return: the chosen candidates(?).
        """

        # create folder and all files from MG probes

        with open(self.probe_path) as probe_fp:
            lines = probe_fp.readlines()
            mg_folder_path, sax_work_directory = self.__create_combined_required_folder_and_files(lines, labels)


        # run new script
        all_mg_files_string = " ".join([os.path.join(mg_folder_path, x) for x in os.listdir(mg_folder_path)])
        full_sax_combined_command = f'{self.SAX_SCRIPT_COMBINED} -s {len(os.listdir(mg_folder_path))} {self.sax_path} {self.rna_path} {all_mg_files_string}'
        print(full_sax_combined_command)
        sax_output = subprocess.run(full_sax_combined_command, shell=True, capture_output=True, cwd=sax_work_directory)
        print(sax_output.stdout)

        # note that mg_paths are numbered according to the line in probe_fp
        score, mg_paths = self.__get_best_scoring_ensemble(sax_work_directory)
        selected_line_indexes = []
        for mg_path in mg_paths:
            selected_line = int(os.path.splitext(mg_path)[0].split('_')[1])
            selected_line_indexes.append(selected_line)
        labels = [0] * len(lines)
        for i in selected_line_indexes:
            labels[i] = 1

        print(f'The lowest scoring ensemble is {score}')
        return labels


    def __get_best_scoring_ensemble(self, sax_work_directory: str):
        """
        Given a string to the sax combination work directory find all files that are an ensemble text file.
        For each of them parse them and return the score and paths of mg probes of the lowest scoring one.
        @param sax_work_directory:
        @return:
        """

        ensemble_files = [file for file in os.listdir(sax_work_directory) if 'ensemble' in file and file.endswith('.txt')]
        ensemble_files.sort(key=lambda x: int(os.path.basename(x).split('_')[2].split('.')[0]))
        best_score = math.inf
        best_file = ""
        mg_paths = []
        scores = []
        best_ensemble_number = 0
        for i, file in enumerate(ensemble_files, start=1):
            cur_score, cur_mg_paths = self.__parse_ensemble_txt(file, sax_work_directory)
            scores.append(cur_score)
            if abs(cur_score - 1) <= abs(best_score - 1):
                best_score = cur_score
                mg_paths = cur_mg_paths
                best_ensemble_number = i
                best_file = file
        print(f'predicted ensemble is of size: {len(mg_paths)}')
        self.__save_best_scoring_ensemble(best_file, best_score)
        if config['plot_sax']:
            self.plot_sax_results(sax_work_directory, best_ensemble_number, scores)
        return best_score, mg_paths

    def __save_best_scoring_ensemble(self, file_name, score):
        """
        Save the best scoring ensemble once found by get_best_scoring_ensamble
        @param file:
        @param score:
        @return:
        """
        scores_dir = os.path.join(self.odir, 'ensemble_scores')
        if not os.path.isdir(scores_dir):
            os.mkdir(scores_dir)
        with open(os.path.join(scores_dir, os.path.basename(file_name)), 'w') as f:
            f.write(file_name + '\n')
            f.write(str(score) + '\n')

    @staticmethod
    def plot_ensemble_scores(sax_work_directory, scores):
        SMALL_SIZE = 16
        MEDIUM_SIZE = 20
        BIGGER_SIZE = 24
        plt.rcParams.update({'axes.facecolor':'white'})
        plt.gcf().subplots_adjust(bottom=0.20)
        plt.gcf().subplots_adjust(left=0.15)
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.1f}')) # 2 decimal places
        plt.grid(color='gray', linewidth=0.2)
        plt.plot(list(range(1, len(scores)+1)), scores)
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(2))
        plt.xlabel('number of $Mg^{2+}$ ions')
        plt.ylabel('$\chi^2$ score')
        plt.title(f'{os.path.basename(config["sax_path"])}, threshold={config["positive_thresh"]}', fontsize=BIGGER_SIZE)
        plt.savefig(os.path.join(sax_work_directory, f'{os.path.basename(config["sax_path"])}_thresh_{config["positive_thresh"]}.png'))
        plt.show()

    @staticmethod
    def plot_sax_results(sax_work_directory, best_ensemble_number, scores):
        SAX.plot_ensemble_scores(sax_work_directory, scores)
        PLOT_SAX_SCRIPT = f"/testing/plotFit1.pl multi_state_model_1_1_1.dat 1 1-Mg-added multi_state_model_{best_ensemble_number}_1_1.dat 2 {best_ensemble_number}-Mg-added"
        print('running '+ PLOT_SAX_SCRIPT)
        subprocess.run(PLOT_SAX_SCRIPT, shell=True, cwd=sax_work_directory)
        img_path = os.path.join(sax_work_directory, "multi_state_model_1_1_1.eps")
        img = mpimg.imread(img_path)
        plt.imshow(img)
        plt.show()

    def __parse_ensemble_txt(self, text_path: str, work_directory:str):
        """
        SAXS combination produces a text file named ensemble_size_x.
        The first line contains the score
        The next x lines contain which mg were used to create the ensemble. Need to extract x+1 first lines and
        parse them.
        @param text_path:
        @return: score and mg ions used.
        """

        FIRST_LINE_SCORE_POS = 1
        MG_LINES_PATH_POS = 2

        number_of_ions = int([text for text in os.path.splitext(text_path)[0].split('_') if text.isdigit()][0])
        with open(os.path.join(work_directory, text_path), 'r') as f:
            mg_paths = []
            first_line = f.readline()
            score = float(first_line.split('|')[FIRST_LINE_SCORE_POS].strip())
            for i in range(number_of_ions):
                cur_line = f.readline()
                mg_paths.append(cur_line.split('|')[MG_LINES_PATH_POS].strip().split('/')[-1].split('.')[0])

        print(f'the score for ensemble {number_of_ions} is {score}')
        return score, mg_paths


class TestMetrics:
    """
    a few random test metrics to run
    dcc - metric of success, for each MG ion see if one of the predicted probes is under some distance threshold to it.
    """

    def __init__(self, predicted_probes_path, mg_path, dcc_thresh=4, dcc_output=None, rna_path=None, **kwargs):
        self.rna_path = rna_path
        self.mg_path = mg_path
        self.predicted_probes_path = predicted_probes_path
        self.dcc_thresh = dcc_thresh
        self.dcc_output = dcc_output

    def run_tests(self):
        self.__dcc()


    def __dcc(self):
        mg_coordinates = extract_atom_coordinates(self.mg_path)
        rna_coordinates = extract_atom_coordinates(self.rna_path)
        relevant_mg_coordinates = []
        for mg_coord in mg_coordinates:
            if find_shortest_dist(mg_coord, rna_coordinates) < config['mg_dist']:
                relevant_mg_coordinates.append(mg_coord)
        try:
            predicted_probe_coordinates = extract_atom_coordinates(self.predicted_probes_path)
        except ValueError as caught_exception:
            print(f'file was empty, no positive predictions made. Exception: {caught_exception}')
            with open(self.dcc_output, 'w') as f:
                f.write(f'{self.predicted_probes_path} 0.0 {len(relevant_mg_coordinates)}\n')
                return
        shortest_distances = []
        for mg_coord in relevant_mg_coordinates:
            shortest_distances.append(find_shortest_dist(mg_coord, predicted_probe_coordinates))
        print(f'the shortest distances are {shortest_distances}')
        dcc_metric = sum(np.array(shortest_distances) <= self.dcc_thresh) / len(shortest_distances)
        print(f'In total the dcc metric for this structure is {dcc_metric} out of {len(relevant_mg_coordinates)} mg atoms')
        with open(self.dcc_output, 'w') as f:
            f.write(f'{self.predicted_probes_path} {dcc_metric} {len(relevant_mg_coordinates)}\n')
