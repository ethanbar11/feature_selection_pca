import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

pd.set_option('display.max_columns', None)


# class ResultsHandler:
#     def __init__(self, ):
#         self.current_dataset = None
#         self.current_algo_name = None
#         self.current_experiment = None
#         self.past_experiments_tot_data = []
#
#         self.datasets = {}
#         # self.filename = filename
#         # self.f = open(filename, 'w')
#         # self.f.write("name, value
#
#     def start_new_experiment(self, dataset, algo_name):
#         self.current_experiment = [('dataset', dataset), ('algo_name', algo_name)]
#         self.current_algo_name = algo_name
#         self.current_dataset = dataset
#         if dataset not in self.datasets:
#             self.datasets[dataset] = {}
#         self.past_experiments_tot_data.append(self.current_experiment)
#
#     def add_result(self, name, value):
#         if name == 'NMI':
#             dataset = self.datasets[self.current_dataset]
#             feature_amount, nmi_value = value
#             if feature_amount not in dataset:
#                 dataset[feature_amount] = {}
#             if self.current_algo_name not in dataset[feature_amount]:
#                 dataset[feature_amount][self.current_algo_name] = []
#             dataset[feature_amount][self.current_algo_name].append(nmi_value)
#         else:
#             self.current_experiment.append((name, value))
#
#     def get_pretty_results(self):
#         lst = []
#         for data in self.past_experiments_tot_data:
#             experiment = {}
#             experiment['dataset_name'] = list(filter(lambda x: x[0] == 'dataset', data))[0][1]
#             experiment['algo_name'] = list(filter(lambda x: x[0] == 'algo_name', data))[0][1]
#             experiment['first_b_precent'] = list(filter(lambda x: x[0] == 'b_precent', data))[0][1]
#             experiment['first_c_precent'] = list(filter(lambda x: x[0] == 'c_precent', data))[0][1]
#             experiment['last_b_precent'] = list(filter(lambda x: x[0] == 'b_precent', data))[-1][1]
#             experiment['last_c_precent'] = list(filter(lambda x: x[0] == 'c_precent', data))[-1][1]
#             experiment['w_accuracy'] = list(filter(lambda x: x[0] == 'w_accuracy', data))[-1][1]
#             dataset_metadata = list(filter(lambda x: x[0] == 'meta dataset', data))[0][1]
#             experiment.update(dataset_metadata)
#             lst.append(experiment)
#         df = pd.DataFrame(lst)
#         return df
#
#     def plot_over_time(self, **kwargs):
#         experiments = {}
#         for experiment in self.past_experiments_tot_data:
#             dataset_identification = list(filter(lambda x: x[0] == 'meta dataset', experiment))[0][1][
#                 kwargs['dataset_identification']]
#             if dataset_identification not in experiments:
#                 experiments[dataset_identification] = []
#             experiments[dataset_identification].append(experiment)
#         for seed, dataset_experiments in experiments.items():
#             fig, ax = plt.subplots()
#             ax.set_title('seed {}'.format(seed))
#             for feature in kwargs['features']:
#                 for experiment in dataset_experiments:
#                     values = list(map(lambda x: x[1], filter(lambda x: x[0] == feature, experiment)))
#                     algo_name = list(filter(lambda x: x[0] == 'algo_name', experiment))[0][1]
#                     label = algo_name + ' ' + feature if len(kwargs['features']) > 1 else algo_name
#                     ax.plot(values, label=label)
#                 # Shrink current axis by 20%
#             box = ax.get_position()
#             ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
#
#             # Put a legend to the right of the current axis
#             ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#             fig.show()
#
#     def plot_nmi(self, **kwargs):
#         for name, dataset in self.datasets.items():
#             for feature_amount, algos in dataset.items():
#                 fig, ax = plt.subplots()
#                 max_value = 0
#                 for algo_name, values in algos.items():
#                     max_value = max(max_value, max(values))
#                     label = algo_name
#                     ax.plot(values, label=label)
#                     # Shrink current axis by 20%
#                 box = ax.get_position()
#                 ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
#                 title = name + ' feature amount: ' + str(feature_amount) + 'max: ' + "{:.3f}".format(float(max_value))
#                 ax.set_title(title)
#
#                 # Put a legend to the right of the current axis
#                 ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#                 fig.show()
#
#     def save_pretty_results(self, filename):
#         pretty_results = self.get_pretty_results()
#         pretty_results.to_csv(filename, index=False)
#
#     def save_raw_results(self, filename):
#         import pickle
#         with open(filename, 'wb') as handle:
#             pickle.dump(self.past_experiments_tot_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


class ResultHandler:
    def __init__(self, device):
        self.current_algo_name = None
        self.current_dataset = None
        self.current_experiment = None
        self.past_experiments_tot_data = {}
        self.device = device

    def set_dataset_name(self, name, X, y, metadata):
        if name not in self.past_experiments_tot_data:
            self.past_experiments_tot_data[name] = {}
        self.current_dataset = self.past_experiments_tot_data[name]
        self.past_experiments_tot_data[name]['X'] = X
        self.past_experiments_tot_data[name]['y'] = y
        self.past_experiments_tot_data[name]['algo_name'] = name
        self.past_experiments_tot_data[name]['meta dataset'] = metadata
        self.past_experiments_tot_data[name]['algos data'] = {}

    def set_algo(self, algo_name):
        self.current_algo_name = algo_name
        if algo_name not in self.current_dataset['algos data']:
            self.current_dataset['algos data'][algo_name] = {}
        self.current_experiment = self.current_dataset['algos data'][algo_name]

    def add_result(self, name, val):
        if name == 'w':
            if 'w' not in self.current_experiment:
                self.current_experiment['w'] = []
            self.current_experiment[name].append(val)

        elif name == 'b_precent' or name == 'c_precent':
            if name not in self.current_experiment:
                self.current_experiment[name] = []
            self.current_experiment[name].append(val)
        else:
            self.current_experiment[name] = val


class SyntheticResultHandler(ResultHandler):
    def set_dataset_name(self, name, X, y, metadata):
        super().set_dataset_name(name, X, y, metadata)
        self.current_dataset['valid_features'] = torch.arange(metadata['n_relevant_features'], device=self.device)

    def add_result(self, name, val):
        super().add_result(name, val)
        if name == 'w':
            w = val
            valid_features = self.current_dataset['valid_features']
            if 'w_accuracy' not in self.current_experiment:
                self.current_experiment['w_accuracy'] = []
            biggest_weight_indices = torch.argsort(w, descending=True)[:len(valid_features)]
            combined = torch.cat((biggest_weight_indices, valid_features))
            uniques, counts = combined.unique(return_counts=True)
            intersection = uniques[counts > 1]
            w_accuracy = len(intersection) / len(valid_features)
            self.current_experiment['w_accuracy'].append(w_accuracy)
            print(w_accuracy)
        elif name == 'b_precent' or name == 'c_precent':
            print(name, val)

    def get_max_result(self):
        return float(max(self.current_experiment['w_accuracy']))


def run_kmeans(X, y, features):
    X = X[:, features]
    n_clusters = len(torch.unique(y))
    times = 5
    random_state = 42
    nmis = []
    for i in range(times):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state + i).fit(X)
        nmi = normalized_mutual_info_score(y, kmeans.labels_)
        nmis.append(nmi)

    return sum(nmis) / len(nmis)


class RealResultsHandler(ResultHandler):
    def __init__(self, device, features_to_check=None):
        super().__init__(device)
        self.features_to_check = features_to_check if features_to_check is not None else [1] + [i for i in
                                                                                                range(40, 200, 20)]

    def plot_experiments_by_feature_amount(self):
        experiments_by_feature_amount = {}
        for dataset_name, dataset_data in self.past_experiments_tot_data.items():
            X = dataset_data['X'].cpu().clone().detach()
            algos = dataset_data['algos data']
            experiments_by_feature_amount[dataset_name] = {}
            max_value_over_dataset =(0,None)
            for feature_amount in self.features_to_check:
                fig, ax = plt.subplots()
                max_value = 0
                for algo_name, algo_data in algos.items():
                    experiments_by_feature_amount[dataset_name][algo_name] = {}
                    w_over_time = algo_data['w']
                    nmi_scores = []
                    for w in w_over_time:
                        top_indices = torch.argsort(w, descending=True)[:feature_amount]
                        nmi_score = run_kmeans(X, dataset_data['y'], top_indices)
                        nmi_scores.append(nmi_score)
                        max_value = max(max_value, max(nmi_scores))
                    label = algo_name
                    ax.plot(nmi_scores, label=label)
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
                title = dataset_name + ' feature amount: ' + str(feature_amount) + 'max: ' + "{:.3f}".format(
                    float(max_value))
                ax.set_title(title)
                if max_value > max_value_over_dataset[0]:
                    max_value_over_dataset = (max_value, title)

                # Put a legend to the right of the current axis
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                fig.show()
            print(max_value_over_dataset)
