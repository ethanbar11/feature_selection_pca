from common import *

from datasets import read_datasets
from baseline_1 import *

import torch


def pca_algorithms_creation(configurations):
    algorithms = []
    for configuration in configurations:
        algorithms.append(PCAFeatureExtraction(**configuration))
    return algorithms


def experiment():
    path = './/results//results_real_datasets.pickle'
    dataset_name = 'synthetic'
    algo_configs = testings
    dataset_config = hard_synthetic_dataset_config_small_feature_amount
    finishing_config = plot_over_time_w
    # Times to perform the experiment repeatably.
    for config in algo_configs:
        config['results_handler'] = results_handler

    algorithms = pca_algorithms_creation(algo_configs)
    for X, y, name, meta in read_datasets(dataset_name, **dataset_config):
        for config, algorithm in zip(algo_configs, algorithms):
            results_handler.start_new_experiment(name, config['algo name'])
            results_handler.add_result('meta dataset', meta)
            print('Running algorithm {} on dataset {}'.format(algorithm, name))
            algorithm.train_wrapper(X, y, meta)
            print(results_handler.current_experiment)

    # finishing_config['function'](**finishing_config)
    results_handler.save_raw_results(path)


if __name__ == '__main__':
    experiment()