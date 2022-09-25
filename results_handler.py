import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

pd.set_option('display.max_columns', None)


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

    def save_results(self, filename):
        import pickle
        with open(filename, 'wb') as handle:
            pickle.dump(self.past_experiments_tot_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
    times = 1
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
            print('Plotting dataset: ', dataset_name)
            algos = dataset_data['algos data']
            experiments_by_feature_amount[dataset_name] = {}
            max_value_over_dataset = (0, None)
            for feature_amount in self.features_to_check:
                print('Plotting feature amount: ', feature_amount)
                fig, ax = plt.subplots()
                max_value = 0
                for algo_name, algo_data in algos.items():
                    print('Calculating algo: ', algo_name)
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

    def print_experiments_by_feature_amount(self):
        experiments_by_feature_amount = {}
        for dataset_name, dataset_data in self.past_experiments_tot_data.items():
            X = dataset_data['X'].cpu().clone().detach()
            print('Plotting dataset: ', dataset_name)
            algos = dataset_data['algos data']
            experiments_by_feature_amount[dataset_name] = {}
            max_value_over_dataset = (0, None)
            for feature_amount in self.features_to_check:
                print('Plotting feature amount: ', feature_amount)
                max_value = 0
                for algo_name, algo_data in algos.items():
                    print('Calculating algo: ', algo_name)
                    experiments_by_feature_amount[dataset_name][algo_name] = {}
                    w_over_time = algo_data['w']
                    nmi_scores = []
                    for w in w_over_time:
                        top_indices = torch.argsort(w, descending=True)[:feature_amount]
                        nmi_score = run_kmeans(X, dataset_data['y'], top_indices)
                        nmi_scores.append(nmi_score)
                        max_value = max(max_value, max(nmi_scores))
                title = dataset_name + ' feature amount: ' + str(feature_amount) + 'max: ' + "{:.3f}".format(
                    float(max_value))
                if max_value > max_value_over_dataset[0]:
                    max_value_over_dataset = (max_value, title)
                print(title)
            print(max_value_over_dataset)
