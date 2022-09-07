"""
Many plotting functions for different situations.
"""

import numpy as no
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
import sklearn.metrics as metrics
import os
import sys


class GraphPlotter:

    def __init__(self, results_path: [str], fig_save_path: str, title: str, legend_labels):
        """
        constructor
        @param predictions:
        @param labels:
        """

        self.predictions = [np.load(os.path.join(base_path, "predictions.npy")) for base_path in results_path]
        self.labels = [np.load(os.path.join(base_path, "labels.npy")) for base_path in results_path]
        self.fig_save_path = fig_save_path
        # self.title = 'ROC Curve 5-fold validation\n with MetalIonRNA Database'
        self.title = title
        self.legend_labels = legend_labels

    def plot_ROC(self):
        SMALL_SIZE = 16
        MEDIUM_SIZE = 20
        BIGGER_SIZE = 24
        VERY_SMALL_SIZE = 12
        plt.rcParams.update({'axes.facecolor':'white'})
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.gcf().subplots_adjust(left=0.15)
        # plt.gcf().subplots_adjust(top=0.8)
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=VERY_SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        plt.grid(color='gray', linewidth=0.1)
        for labels, predictions, legend_labels in zip(self.labels, self.predictions, self.legend_labels):
            fpr, tpr, _ = metrics.roc_curve(labels, predictions)
            auc = metrics.roc_auc_score(labels, predictions)
            plt.plot(fpr, tpr, label=f"{legend_labels}, auc={auc:.3f}")
        plt.legend(loc=4)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title(self.title, fontsize=BIGGER_SIZE)
        plt.savefig(os.path.join(self.fig_save_path, self.title))
        plt.show()


def main():
    fig_path = "test_figures"
    results_path_base = "test_results"
    results_path = [os.path.join(results_path_base, result) for result in sys.argv[1].split()]
    title = sys.argv[2]
    legend_labels = sys.argv[3].split(",")
    plotter = GraphPlotter(results_path, fig_path, title, legend_labels)
    plotter.plot_ROC()


if __name__ == '__main__':
    main()