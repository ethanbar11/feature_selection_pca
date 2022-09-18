import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import torch
from sklearn.metrics import normalized_mutual_info_score
import sklearn.cluster

pd.set_option('display.max_columns', None)


class ResultsHandler:
    def __init__(self, ):
        self.current_dataset = None
        self.current_algo_name = None
        self.current_experiment = None
        self.past_experiments_tot_data = []

        self.datasets = {}
        # self.filename = filename
        # self.f = open(filename, 'w')
        # self.f.write("name, value

    def start_new_experiment(self, dataset, algo_name):
        self.current_experiment = [('dataset', dataset), ('algo_name', algo_name)]
        self.current_algo_name = algo_name
        self.current_dataset = dataset
        if dataset not in self.datasets:
            self.datasets[dataset] = {}
        self.past_experiments_tot_data.append(self.current_experiment)

    def add_result(self, name, value):
        if name == 'NMI':
            dataset = self.datasets[self.current_dataset]
            feature_amount, nmi_value = value
            if feature_amount not in dataset:
                dataset[feature_amount] = {}
            if self.current_algo_name not in dataset[feature_amount]:
                dataset[feature_amount][self.current_algo_name] = []
            dataset[feature_amount][self.current_algo_name].append(nmi_value)
        else:
            self.current_experiment.append((name, value))

    def get_pretty_results(self):
        lst = []
        for data in self.past_experiments_tot_data:
            experiment = {}
            experiment['dataset_name'] = list(filter(lambda x: x[0] == 'dataset', data))[0][1]
            experiment['algo_name'] = list(filter(lambda x: x[0] == 'algo_name', data))[0][1]
            experiment['first_b_precent'] = list(filter(lambda x: x[0] == 'b_precent', data))[0][1]
            experiment['first_c_precent'] = list(filter(lambda x: x[0] == 'c_precent', data))[0][1]
            experiment['last_b_precent'] = list(filter(lambda x: x[0] == 'b_precent', data))[-1][1]
            experiment['last_c_precent'] = list(filter(lambda x: x[0] == 'c_precent', data))[-1][1]
            experiment['w_accuracy'] = list(filter(lambda x: x[0] == 'w_accuracy', data))[-1][1]
            dataset_metadata = list(filter(lambda x: x[0] == 'meta dataset', data))[0][1]
            experiment.update(dataset_metadata)
            lst.append(experiment)
        df = pd.DataFrame(lst)
        return df

    def plot_over_time(self, **kwargs):
        experiments = {}
        for experiment in self.past_experiments_tot_data:
            dataset_identification = list(filter(lambda x: x[0] == 'meta dataset', experiment))[0][1][
                kwargs['dataset_identification']]
            if dataset_identification not in experiments:
                experiments[dataset_identification] = []
            experiments[dataset_identification].append(experiment)
        for seed, dataset_experiments in experiments.items():
            fig, ax = plt.subplots()
            ax.set_title('seed {}'.format(seed))
            for feature in kwargs['features']:
                for experiment in dataset_experiments:
                    values = list(map(lambda x: x[1], filter(lambda x: x[0] == feature, experiment)))
                    algo_name = list(filter(lambda x: x[0] == 'algo_name', experiment))[0][1]
                    label = algo_name + ' ' + feature if len(kwargs['features']) > 1 else algo_name
                    ax.plot(values, label=label)
                # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            fig.show()

    def plot_nmi(self, **kwargs):
        for name, dataset in self.datasets.items():
            for feature_amount, algos in dataset.items():
                fig, ax = plt.subplots()
                max_value = 0
                for algo_name, values in algos.items():
                    max_value = max(max_value, max(values))
                    label = algo_name
                    ax.plot(values, label=label)
                    # Shrink current axis by 20%
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
                title = name + ' feature amount: ' + str(feature_amount) +'max: ' + "{:.3f}".format(float(max_value))
                ax.set_title(title)

                # Put a legend to the right of the current axis
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                fig.show()

    def save_pretty_results(self, filename):
        pretty_results = self.get_pretty_results()
        pretty_results.to_csv(filename, index=False)

    def save_raw_results(self, filename):
        import pickle
        with open(filename, 'wb') as handle:
            pickle.dump(self.past_experiments_tot_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Running k-means and returning accuracy based on algo.
def run_algo(algo, X, y, seed=42, feature_amount=None):
    NUM_OF_RUNS = 1
    features = algo.get_relevant_features(X, feature_amount)
    n_clusters = len(torch.unique(y))
    X_used = X[:, features]
    accuracies = []
    for i in range(NUM_OF_RUNS):
        kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=seed + i).fit(X_used)
        y_pred = kmeans.labels_
        mutual_info_score = normalized_mutual_info_score(y, y_pred)
        accuracies.append(mutual_info_score)
    return torch.mean(torch.tensor(accuracies))


# Algo Configs

default_algo_config = {'n_components': 50,
                       'use_loss_B': False,
                       'use_normalization': True,
                       'accumulating_w': False,
                       'norm': 1,
                       'xi': 0.01,
                       'iterative': False,
                       'learning_rate': 0.003,
                       'epochs': 81,
                       'pca_only_on_true_features': False,
                       'algo name': 'Default'}

debugging_config = default_algo_config.copy()
debugging_config['use_fake_groups'] = True
debugging_config['pca_only_on_true_features'] = False
debugging_config['iterative'] = False
debugging_config['use_loss_C'] = True
debugging_config['use_loss_B'] = False
debugging_config['use_loss_ratio'] = False
debugging_config['use_normalization'] = True
debugging_config['algo name'] = 'Only C Loss'

testings = []
testing_basic_config = default_algo_config.copy()
testing_basic_config['use_fake_groups'] = True
testing_basic_config['pca_only_on_true_features'] = False
testing_basic_config['iterative'] = True
testing_basic_config['use_loss_C'] = True
testing_basic_config['use_loss_B'] = False
testing_basic_config['use_loss_ratio'] = False
testing_basic_config['use_normalization'] = True
testing_basic_config['algo name'] = 'Only C Loss'
testing_basic_config['feature_amount'] = [1] + [i for i in range(40, 201, 20)]

for n_components in [1, 3, 5, 10, 20, 50, 100]:
    for use_normalization in [False, True]:
        testing_basic_config['use_normalization'] = use_normalization
        testing_basic_config['n_components'] = n_components
        testing_basic_config['algo name'] = 'k: {}, norm: {}'.format(n_components, use_normalization)
        testings.append(testing_basic_config.copy())

# Dataset Configs

default_synthetic_dataset_config = {'seed': 65, 'times': 1}

hard_synthetic_dataset_config = default_synthetic_dataset_config.copy()
hard_synthetic_dataset_config['times'] = 1
hard_synthetic_dataset_config['SD'] = 0.2
hard_synthetic_dataset_config['n_classes'] = 3

hard_synthetic_dataset_config_small_feature_amount = hard_synthetic_dataset_config.copy()
hard_synthetic_dataset_config_small_feature_amount['n_relevant_features'] = 100
hard_synthetic_dataset_config_small_feature_amount['n_false_feature'] = 100

real_datasets = {}

# Finishing Configs

results_handler = ResultsHandler()


def print_results(**kwargs):
    print(results_handler.get_pretty_results())


print_weights_only = {'function': print_results}
plot_over_time_c = {'function': results_handler.plot_over_time, 'features': ['c_precent']}
plot_over_time_w = {'function': results_handler.plot_over_time, 'features': ['w_accuracy']}
plot_over_time_k_means_nmi = {'function': results_handler.plot_nmi, 'features': ['NMI'],
                              'dataset_identification': 'feature_amount'}
plot_over_time_b = {'function': results_handler.plot_over_time, 'feature': 'b_precent'}
