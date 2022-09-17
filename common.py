import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)


class ResultsHandler:
    def __init__(self, ):
        self.current_experiment = None
        self.past_experiments = []
        # self.filename = filename
        # self.f = open(filename, 'w')
        # self.f.write("name, value

    def start_new_experiment(self, dataset, algo_name):
        self.current_experiment = [('dataset', dataset), ('algo_name', algo_name)]
        self.past_experiments.append(self.current_experiment)

    def add_result(self, name, value):
        self.current_experiment.append((name, value))

    def get_pretty_results(self):
        lst = []
        for data in self.past_experiments:
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
        for experiment in self.past_experiments:
            seed_dataset = list(filter(lambda x: x[0] == 'meta dataset', experiment))[0][1]['seed']
            if seed_dataset not in experiments:
                experiments[seed_dataset] = []
            experiments[seed_dataset].append(experiment)
        for seed, dataset_experiments in experiments.items():
            fig, ax = plt.subplots()
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

    def save_pretty_results(self, filename):
        pretty_results = self.get_pretty_results()
        pretty_results.to_csv(filename, index=False)

    def save_raw_results(self, filename):
        import pickle
        with open(filename, 'wb') as handle:
            pickle.dump(self.past_experiments, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Algo Configs

default_algo_config = {'n_components': 1,
                       'use_loss_B': False,
                       'use_normalization': True,
                       'norm': 1,
                       'xi': 0.01,
                       'iterative': False,
                       'learning_rate': 0.01,
                       'epochs': 20,
                       'algo name': 'Default'}

iterative_config = default_algo_config.copy()
iterative_config['iterative'] = True
iterative_config['accumulating_w'] = True
iterative_config['algo name'] = 'Iterative norm 1'

testing_iterative = iterative_config.copy()
testing_iterative['algo name'] = 'testing iterative'
testing_iterative['epochs'] = 80
testing_iterative['learning_rate'] = 0.1
testing_iterative['norm'] = 1
testing_iterative['accumulating_w'] = False
testing_iterative['iterative'] = False
testing_iterative['easy_accumulation'] = False

testings = []
for k in [1, 3, 5, 7]:
    non_iterative = testing_iterative.copy()
    non_iterative['iterative'] = False
    non_iterative['algo name'] = 'non iterative k = ' + str(k)
    yes_iterative = testing_iterative.copy()
    yes_iterative['iterative'] = True
    yes_iterative['algo name'] = 'iterative k = ' + str(k)
    accumulating_iterative = yes_iterative.copy()
    accumulating_iterative['accumulating_w'] = True
    accumulating_iterative['algo name'] = 'accumulating iterative k = ' + str(k)
    easy_accumulating_iterative = accumulating_iterative.copy()
    easy_accumulating_iterative['easy_accumulation'] = True
    easy_accumulating_iterative['algo name'] = 'easy accumulating iterative k = ' + str(k)
    testings.append(non_iterative)
    testings.append(yes_iterative)
    testings.append(accumulating_iterative)
    testings.append(easy_accumulating_iterative)

# Dataset Configs

default_synthetic_dataset_config = {'seed': 42, 'times': 50}

hard_synthetic_dataset_config = default_synthetic_dataset_config.copy()
hard_synthetic_dataset_config['times'] = 10
hard_synthetic_dataset_config['SD'] = 0.2
hard_synthetic_dataset_config['n_classes'] = 2

hard_synthetic_dataset_config_small_feature_amount = hard_synthetic_dataset_config.copy()
hard_synthetic_dataset_config_small_feature_amount['n_relevant_features'] = 10
hard_synthetic_dataset_config_small_feature_amount['n_false_feature'] = 10
# Finishing Configs

results_handler = ResultsHandler()


def print_results(**kwargs):
    print(results_handler.get_pretty_results())


print_weights_only = {'function': print_results}
plot_over_time_c = {'function': results_handler.plot_over_time, 'features': ['c_precent']}
plot_over_time_w = {'function': results_handler.plot_over_time, 'features': ['w_accuracy']}
plot_over_time_b = {'function': results_handler.plot_over_time, 'feature': 'b_precent'}
