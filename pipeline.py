from ResultsHandler import SyntheticResultHandler, RealResultsHandler
from common import *

from datasets import read_datasets
from pca_feature_selection import *

import torch

algos = {
    'PCAFeatureExtraction': PCAFeatureExtraction,
    'IterativePCAFeatureExtraction': IterativePCAFeatureExtraction,
}


def pca_algorithms_creation(X, y=None, meta=None, configurations=None):
    algorithms = []
    for configuration in configurations:
        algo_class = algos[configuration['algo class']]
        algorithms.append(algo_class(X, y, meta, **configuration))
    return algorithms


def experiment():
    path = './/results//results_real_datasets.pickle'
    dataset_name = 'synthetic'
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print('Running on device: ', device)
    algo_configs = [debugging_config]
    dataset_config = hard_synthetic_dataset_config_small_feature_amount
    results_handler = SyntheticResultHandler(device)

    finish_func = lambda: print('Woho')
    # Times to perform the experiment repeatably.
    for algo_config in algo_configs:
        algo_config['results_handler'] = results_handler
        algo_config['device'] = device

    for X, y, name, meta in read_datasets(dataset_name, **dataset_config):
        X = X.to(device)
        algorithms = pca_algorithms_creation(X, y, meta, algo_configs)
        results_handler.set_dataset_name(name, X, y, meta)
        for algo_config, algorithm in zip(algo_configs, algorithms):
            print('Running algorithm {} on dataset {}'.format(algo_config['algo name'], name))
            results_handler.set_algo(algo_config['algo name'])
            algorithm.train()
            # max_result = results_handler.get_max_result()
            # print('Finished algorithm {} on dataset {}, max result is :{}'.format(algo_config['algo name'], name,
            #                                                                       max_result))
    finish_func()


if __name__ == '__main__':
    experiment()
