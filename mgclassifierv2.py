import argparse
import os

from utils import fix_dict_in_config
from config import base_config
from assests import AssetManager
from preprocessing.train_test_split import TrainTestSplit
import preprocessing.dataset as dataset
import torch_geometric.loader as geom_loader
from preprocessing import dataset as ds
from models.training import GraphGNN
from inference.inferencePipeline import InferencePipeline
import wandb
from preprocessing.kfold_creator import Kfold
from preprocessing.preprocess_utils import create_random_sampler, evaluate_dataset
from testing.test_utils import save_test_results
from inference.scoper_pipeline import SCOPER



def preprocess(args, kfold_path=base_config['kfold_path']):
    assets = AssetManager(args.base_dir)
    dataset = ds.get_dataset(args.dataset_id, args.dataset_path, kfold_path)
    return dataset


def split_samples(args):
    """
    This function splits the database into training validation and test sets according to high enough thresholds that
    even allow such a split according to a similarity matrix that was calculated over the entire database.
    We first randomly create the test set and then with the rest of the samples we add them to the training set if
    they aren't too similar to samples inside the test set according to the similarity matrix.
    Things that are too similar are added to the validation set.

    If successful the split is written to a file and saved.
    """
    train_test_split = TrainTestSplit(args.base_dir, args.split_regime, args.similarity_matrix, args.train_percentage,
                                      args.k, args.threshold, args.out_dir)
    train_test_split.split_selection()


def train(args):
    config = dict()
    config.update(base_config)
    run = wandb.init(project=config['wandb_dict']['project_name'], config=config)
    if run and run.name != None:
        wandb.config["run_name"] = run.name
        config["run_name"] = run.name
    if args.sweeps:
        fix_dict_in_config(wandb)
        config = wandb.config
    dataset = preprocess(args, config["kfold_path"])
    train_data, val_data, test_data = dataset.train_val_test_split(config['split_mode'])

    # get some information about the datasets
    evaluate_dataset(train_data, 'train_data')
    evaluate_dataset(val_data, 'val_data')
    evaluate_dataset(test_data, 'test_data')
    if not args.keep_transform:
        print('removing transform')
        val_data.transform = None
        test_data.transform = None
    if args.kfold:
        random_sampler = create_random_sampler(train_data)
        train_graph_loader = geom_loader.DataLoader(train_data, config['train_dict']['batch_size'], num_workers=config['train_dict']['num_workers'], sampler=random_sampler)
    else:
        train_graph_loader = geom_loader.DataLoader(train_data, config['train_dict']['batch_size'], num_workers=config['train_dict']['num_workers'], shuffle=True)
    val_graph_loader = geom_loader.DataLoader(val_data, config['train_dict']['batch_size'], num_workers=config['train_dict']['num_workers'])
    test_graph_loader = geom_loader.DataLoader(test_data, config['train_dict']['batch_size'], num_workers=config['train_dict']['num_workers'])

    model = GraphGNN(config)
    single_node = "SINGLE_NODE" in config['model_name']

    if config['test']:
        model.train(train_graph_loader, val_graph_loader, single_node)
        # for a k-fold validation the test will be the validation set.
        print("training completed, beginning test on the test set.")
        model.test(test_graph_loader, config['test_dict']['thresh'], single_node, inference=True)
    else:
        # this is done for kfold validation where there is no validation set.
        model.train(train_graph_loader, test_graph_loader, single_node)
    wandb.finish()

def test(args):
    config = base_config
    model_path = args.model_path
    model_config_path = args.model_config_path
    model = GraphGNN(base_config)
    print('loading model')
    model.load(model_path, 'eval', model_config_path)
    print('getting dataset')
    dataset = ds.get_dataset(args.dataset_id, args.dataset_path, config["kfold_path"])
    print('splitting dataset')
    train_data, val_data, test_data = dataset.train_val_test_split(config['split_mode'])
    val_data.transform = None
    test_data.transform = None
    train_graph_loader = geom_loader.DataLoader(train_data, config['train_dict']['batch_size'], num_workers=config['train_dict']['num_workers'], shuffle=True)
    val_graph_loader = geom_loader.DataLoader(val_data, config['train_dict']['batch_size'], num_workers=config['train_dict']['num_workers'])
    test_graph_loader = geom_loader.DataLoader(test_data, config['train_dict']['batch_size'], num_workers=config['train_dict']['num_workers'])
    predictions, labels = model.test(test_graph_loader, base_config['test_dict']['thresh'], inference=base_config['inference'])
    print(predictions)
    save_test_results(args.predictions_path, model_path, predictions, labels)


def inference(args):
    fpath = args.file_path
    odir = args.output_dir
    test = args.test
    cleanup = args.cleanup
    inference_type = args.inference_type
    model_path = args.model_path
    config_path = args.config_path
    pymol = args.pymol
    dcc_output = args.dcc_output
    overwrite = args.overwrite
    foxs_script = args.foxs_script
    multifoxs_script = args.multifoxs_script
    pipeline = InferencePipeline(odir, fpath, inference_type, model_path, config_path, pymol, overwrite, foxs_script, multifoxs_script)
    kwargs = pipeline.infer()
    kwargs['dcc_output'] = dcc_output
    if test:
        pipeline.test(**kwargs)
    if cleanup:
        pipeline.cleanup()


def scoper(args):
    """
    Runs full scoper pipeline
    :param args:
    :return:
    """

    fpath = args.file_path
    base_dir = args.base_dir
    saxs_path = args.saxs_path
    saxs_script_path = args.foxs_script
    kgs_k = args.kgs_k
    inference_type = args.inference_type
    model_path = args.model_path
    config_path = args.config_path
    multifoxs_script_path = args.multifoxs_script
    addhydrogens_script_path = args.addhydrogens_script

    scoper = SCOPER(fpath, saxs_path, base_dir, inference_type,
                    model_path, config_path,
                    saxs_script_path, multifoxs_script_path, addhydrogens_script_path, kgs_k)
    kwargs = scoper.run()


def kfold_validation(args):
    """
    This function is responsible for creating k folds (files with names of pdbs in train and test)
    For each fold a model is trained
    In the end all models are evaluated and their average test scores are recorded.
    @param args:
    @return:
    """
    # create k folds in new files
    kfold = Kfold(args.kfold_dir, args.number_of_folds)
    kfold(args.dataset_path)
    # call train with each kfold file generated on bio08
    for fold_file in os.listdir(kfold.kfold_dir):
        print(f'beginning fold {fold_file}')
        base_config['kfold_path'] = os.path.join(kfold.kfold_dir, fold_file)
        train(args)


    # evaluate all models when they finish


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bd', '--base-dir', type=str, required=True)
    parser.add_argument('-kfp', '--kfold-path', type=str, required=False, default=None)

    action_parsers = parser.add_subparsers(dest='action')
    action_parsers.required = True

    preprocess_parser = action_parsers.add_parser('preprocess')
    preprocess_parser.add_argument('-di', '--dataset-id', type=str, choices=dataset.supported_datasets, required=True)
    preprocess_parser.add_argument('-dp', '--dataset-path', type=str, required=False)
    preprocess_parser.set_defaults(func=preprocess)

    split_samples_parser = action_parsers.add_parser('split-samples')
    split_samples_parser.add_argument('-splt', '--split-regime', type=int, required=True)
    split_samples_parser.add_argument('-sim', '--similarity-matrix', type=str, required=True)
    split_samples_parser.add_argument('-tp', '--train-percentage', type=float, required=True)
    split_samples_parser.add_argument('-th', '--threshold', type=float, required=True)
    split_samples_parser.add_argument('-k', '--k', type=int, required=True)
    split_samples_parser.add_argument('-od', '--out-dir', type=str, required=True)
    split_samples_parser.set_defaults(func=split_samples)

    train_parser = action_parsers.add_parser('train')
    train_parser.add_argument('-di', '--dataset-id', type=str, choices=dataset.supported_datasets, required=True)
    train_parser.add_argument('-mn', '--model-name', type=str, required=True)
    train_parser.add_argument('-dp', '--dataset-path', type=str, required=False)
    train_parser.add_argument('-cp', '--checkpoint-path', type=str, required=False)
    train_parser.add_argument('-s', '--sweeps', type=bool, default=False)  # to add bool simply add the flag with any text underneath
    train_parser.add_argument('-kt', '--keep-transform', type=bool, default=True, required=False)
    train_parser.add_argument('-kf', '--kfold', type=bool, default=False, required=False)
    train_parser.set_defaults(func=train)

    test_parser = action_parsers.add_parser('test')
    test_parser.add_argument('-di', '--dataset-id', type=str, choices=dataset.supported_datasets, required=True)
    test_parser.add_argument('-mn', '--model-name', type=str, required=True)
    test_parser.add_argument('-mp', '--model-path', type=str, required=True)
    test_parser.add_argument('-mcp', '--model-config-path', type=str, required=True)
    test_parser.add_argument('-dp', '--dataset-path', type=str, required=False)
    test_parser.add_argument('-cp', '--checkpoint-path', type=str, required=False)
    test_parser.add_argument('-s', '--sweeps', type=bool, default=False)  # to add bool simply add the flag with any text underneath
    test_parser.add_argument('-pp', '--predictions-path', type=str, default=None, required=True)
    test_parser.set_defaults(func=test)

    inference_parser = action_parsers.add_parser('inference')
    inference_parser.add_argument('-o', '--output-dir', type=str,  required=True)
    inference_parser.add_argument('-fp', '--file-path', type=str, required=True)
    inference_parser.add_argument('-it', '--inference-type', type=str, required=True)
    inference_parser.add_argument('-mp', '--model-path', type=str, required=True)
    inference_parser.add_argument('-do', '--dcc-output', type=str, required=True)
    inference_parser.add_argument('-cp', '--config-path', type=str, required=False, default=None)
    inference_parser.add_argument('-pymol', '--pymol', type=bool, required=False, default=False)
    inference_parser.add_argument('-ov', '--overwrite', type=bool, required=False, default=False)
    inference_parser.add_argument('-t', '--test', type=bool, required=False, default=False)
    inference_parser.add_argument('-cu', '--cleanup', type=bool, required=False, default=False)
    inference_parser.add_argument('-fs', '--foxs-script', type=str, required=True, default=False)
    inference_parser.add_argument('-mfs', '--multifoxs-script', type=str, required=True, default=False)
    inference_parser.set_defaults(func=inference)

    scoper_parser = action_parsers.add_parser('scoper')
    scoper_parser.add_argument('-fp', '--file-path', type=str,  required=True)
    scoper_parser.add_argument('-sp', '--saxs-path', type=str,  required=True)
    scoper_parser.add_argument('-mp', '--model-path', type=str, required=True)
    scoper_parser.add_argument('-cp', '--config-path', type=str, required=False, default=None)
    scoper_parser.add_argument('-it', '--inference-type', type=str, required=True)
    scoper_parser.add_argument('-fs', '--foxs-script', type=str, required=True, default=False)
    scoper_parser.add_argument('-mfs', '--multifoxs-script', type=str, required=True, default=False)
    scoper_parser.add_argument('-ahs', '--addhydrogens-script', type=str, required=True, default=False)
    scoper_parser.add_argument('-kk', '--kgs-k', type=int, required=False, default=1)
    scoper_parser.set_defaults(func=scoper)

    kfold_parser = action_parsers.add_parser('kfold')
    kfold_parser.add_argument('-k', '--number-of-folds', type=int, required=True, default=5)
    kfold_parser.add_argument('-kdir','--kfold-dir', type=str, required=True)
    kfold_parser.add_argument('-di', '--dataset-id', type=str, choices=dataset.supported_datasets, required=True)
    kfold_parser.add_argument('-mn', '--model-name', type=str, required=True)
    kfold_parser.add_argument('-dp', '--dataset-path', type=str, required=False)
    kfold_parser.add_argument('-cp', '--checkpoint-path', type=str, required=False)
    kfold_parser.add_argument('-s', '--sweeps', type=bool, default=False)  # to add bool simply add the flag with any text underneath
    kfold_parser.add_argument('-kf', '--kfold', type=bool, default=True, required=True)
    kfold_parser.add_argument('-kt', '--keep-transform', type=bool, default=True, required=True)
    kfold_parser.set_defaults(func=kfold_validation)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()