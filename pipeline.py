from common import ResultsHandler
from datasets import read_datasets
from baseline_1 import *
import torch


def pca_algorithms_creation(configurations):
    algorithms = []
    for configuration in configurations:
        algorithms.append(PCAFeatureExtraction(**configuration))
    return algorithms


configurations = []

default_algo_config = {'n_components': 5,
                       'use_loss_B': False,
                       'use_normalization': True,
                       'norm': 1,
                       'xi': 0.01,
                       'iterative': False,
                       'epochs': 100,
                       'algo name': 'Default'}

for iterative in [False, True]:
    for use_normalization in [False, True]:
        for norm in [1, 2]:
            config = default_algo_config.copy()
            config['iterative'] = iterative
            config['use_normalization'] = use_normalization
            config['norm'] = norm
            config['algo name'] = 'iterative={}, use_normalization={}, norm={}'.format(iterative, use_normalization,
                                                                                       norm)
            configurations.append(config)

default_synthetic_dataset_config = {'seed': 42, 'times': 2}


def experiment():
    path = './/results//results.csv'
    dataset_name = 'synthetic'
    results_handler = ResultsHandler()
    # Times to perform the experiment repeatably.
    for config in configurations:
        config['results_handler'] = results_handler

    algorithms = pca_algorithms_creation(configurations)
    for X, y, name, meta in read_datasets(dataset_name, **default_synthetic_dataset_config):
        for config, algorithm in zip(configurations, algorithms):
            results_handler.start_new_experiment(name, config['algo name'])
            print('Running algorithm {} on dataset {}'.format(algorithm, name))
            algorithm.train_wrapper(X, y, meta)
            print(results_handler.current_experiment)

    print(results_handler.get_pretty_results())
    results_handler.save_results(path)


if __name__ == '__main__':
    experiment()
