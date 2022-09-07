"""
Util functions for mgclassifier test
"""

import os
import numpy as np



def save_test_results(test_result_dir: str, model_name: str, predictions: [int], labels: [int]):
    """
    Create directory for test results according to base test_result_dir and model name
    proceed to save predictions and labels.
    @param test_result_dir:
    @param model_name:
    @param predictions:
    @param labels:
    @return:
    """

    model_results_dir = os.path.join(test_result_dir, os.path.basename(model_name))
    if not os.path.isdir(model_results_dir):
        os.mkdir(model_results_dir)
    np.save(os.path.join(model_results_dir, "predictions.npy"), np.array(predictions))
    np.save(os.path.join(model_results_dir, "labels.npy"), np.array(labels))
