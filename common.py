import sklearn
import torch
from sklearn.metrics import normalized_mutual_info_score
import sklearn.cluster


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

default_algo_config = {'n_components': 7,
                       'use_loss_B': False,
                       'use_normalization': True,
                       'accumulating_w': False,
                       'norm': 1,
                       'xi': 0.01,
                       'iterative': False,
                       'learning_rate': 0.3,
                       'epochs': 200,
                       'pca_only_on_true_features': False,
                       'algo name': 'Default',
                       'use_clamping': False,
                       'algo class': 'PCAFeatureExtraction'}

debugging_config = default_algo_config.copy()
debugging_config['use_fake_groups'] = True
debugging_config['pca_only_on_true_features'] = False
debugging_config['use_loss_B'] = False
debugging_config['use_loss_C'] = True
debugging_config['use_loss_ratio'] = True
debugging_config['normalize_data'] = True
debugging_config['use_normalization'] = False
debugging_config['use_clamping'] = False
debugging_config['algo name'] = 'Only C Loss'
debugging_config['algo class'] = 'PCAFeatureExtraction'
debugging_config['update_P'] = False

testings = []
testing_basic_config = default_algo_config.copy()
testing_basic_config['use_fake_groups'] = True
testing_basic_config['pca_only_on_true_features'] = False
testing_basic_config['iterative'] = False
testing_basic_config['use_loss_C'] = True
testing_basic_config['use_loss_B'] = False
testing_basic_config['normalize_data'] = True
testing_basic_config['use_loss_ratio'] = True

testing_basic_config['use_normalization'] = False
testing_basic_config['algo name'] = 'Only C Loss'
testing_basic_config['update_P'] = True

for n_components in [5, 10, 50, 80, 120, 150]:
    for use_loss_ratio in [True, False]:
        testing_basic_config = testing_basic_config.copy()
        testing_basic_config['n_components'] = n_components
        testing_basic_config['use_loss_ratio'] = use_loss_ratio
        testing_basic_config['algo name'] = 'k={}, ratio={}'.format(n_components, use_loss_ratio)
        testings.append(testing_basic_config)

# Dataset Configs

default_synthetic_dataset_config = {'seed': 42, 'times': 1}

hard_synthetic_dataset_config = default_synthetic_dataset_config.copy()
hard_synthetic_dataset_config['times'] = 1
hard_synthetic_dataset_config['SD'] = 0.2
hard_synthetic_dataset_config['n_classes'] = 2

hard_synthetic_dataset_config_small_feature_amount = hard_synthetic_dataset_config.copy()
hard_synthetic_dataset_config_small_feature_amount['n_relevant_features'] = 10
hard_synthetic_dataset_config_small_feature_amount['n_false_feature'] = 50

real_datasets = {}

# Finishing Configs


# def print_results(**kwargs):
#     print(results_handler.get_pretty_results())
#
#
# print_weights_only = {'function': print_results}
# plot_over_time_c = {'function': results_handler.plot_over_time, 'features': ['c_precent']}
# plot_over_time_w = {'function': results_handler.plot_over_time, 'features': ['w_accuracy']}
# plot_over_time_k_means_nmi = {'function': results_handler.plot_nmi, 'features': ['NMI'],
#                               'dataset_identification': 'feature_amount'}
# plot_over_time_b = {'function': results_handler.plot_over_time, 'feature': 'b_precent'}
