from torch import autograd
import os
import logging

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.optim
import sklearn.cluster
from sklearn.metrics.cluster import normalized_mutual_info_score

import datasets
import synthetic_data_generator

logging.basicConfig(level=logging.INFO, format='%(message)s')


def log_info(*args):
    logging.info(args_to_string(args))


def log_debug(*args):
    logging.debug(args_to_string(args))


# Convert args to one string
def args_to_string(args):
    return ' '.join([str(arg) for arg in args])


class FeatureExtractionAlgorithm:
    def __init__(self, **kwargs):
        pass

    # Expecting X to be numpy array of size (n_samples, n_features)
    def get_relevant_features(self, X):
        raise NotImplementedError

    def train(self, *kwargs):
        pass


class Baseline(FeatureExtractionAlgorithm):
    def get_relevant_features(self, X, amount=0):
        return [i for i in range(amount)]

    def __str__(self):
        return 'Baseline'


# This is specific to the synthetic data generator
class Perfect(FeatureExtractionAlgorithm):
    def get_relevant_features(self, X):
        return [i for i in range(X.shape[1]) if i < 5]


def calculate_compression(u, v, P):
    diff = u - v
    outcome = torch.linalg.norm(torch.matmul(diff, P.T), dim=1) / torch.linalg.norm(diff, dim=1)
    return outcome


def calculate_compression_w(u, v, P, w):
    weighted_diff = w * (u - v)
    weighted_diff_P = torch.matmul(weighted_diff, P.T)
    outcome = torch.linalg.norm(weighted_diff_P, dim=1) / torch.linalg.norm(weighted_diff, dim=1)
    return outcome


class Optimizer:
    def __init__(self, B, C, **kwargs):
        self.use_loss_B = kwargs['use_loss_B']
        self.use_normalization = kwargs['use_normalization']
        self.use_clamping = kwargs['use_clamping'] if 'use_clamping' in kwargs else True
        self.valid_features = kwargs['valid_features']
        self.BATCH_SIZE = int(1e4)
        self.LEARNING_RATE = kwargs['learning_rate']
        self.norm = kwargs['norm']
        self.results_handler = kwargs['results_handler']
        self.set_vals(B, C, np.zeros((1, 1)))
        log_debug("Starting to train w. Number of batches: ", len(self.B_batches))
        self.w = torch.ones((B.shape[-1]), requires_grad=True)
        self.current_loss_value = None
        self.optimizer = torch.optim.SGD(params=[self.w], lr=self.LEARNING_RATE)

    # You have to call for set_vals before using.
    def set_vals(self, B, C, P):
        self.B = B
        self.C = C
        self.P = P
        self.B_batches = torch.split(B, self.BATCH_SIZE, dim=0)
        self.C_batches = torch.split(C, self.BATCH_SIZE, dim=0)

    def optimize_w(self, epochs):
        # B is a matrix of size (chi * n**2 , d)
        # w is a vector of size (d, 1)
        # P is a matrix of size (k, d)
        # Split B into batches of rows
        for epoch in range(epochs):

            for batch_b, batch_c in zip(self.B_batches, self.C_batches):
                # Calculating the gradient of the loss function
                # with respect to w
                self.optimizer.zero_grad()

                u = batch_b[:, 0, :]
                v = batch_b[:, 1, :]
                u2 = batch_c[:, 0, :]
                v2 = batch_c[:, 1, :]

                loss_C = torch.mean(calculate_compression_w(u2, v2, self.P, self.w)) * (-1.0)
                loss = loss_C
                if self.use_loss_B:
                    loss_B = torch.mean(calculate_compression_w(u, v, self.P, self.w))
                    loss += loss_B

                loss.backward()
                self.current_loss_value = loss.item()
                self.optimizer.step()
                if self.use_normalization:
                    # Normalization of 2 order
                    self.w.data = self.w.data / torch.linalg.norm(self.w.data, ord=self.norm)
                    # Softmax normalization
                    # self.w.data = torch.nn.functional.softmax(self.w.data, dim=0)
                if self.use_clamping:
                    self.w.data = torch.clamp(self.w.data, min=0.0, max=1.0)
                if epoch != 0:
                    self.print_status()
        return self.w.detach()

    def print_status(self, w=None):
        if w is None:
            w = self.w
        if self.valid_features is not None:
            biggest_weight_indices = torch.argsort(w, descending=True)[:len(self.valid_features)]
            combined = torch.cat((biggest_weight_indices, self.valid_features))
            uniques, counts = combined.unique(return_counts=True)
            intersection = uniques[counts > 1]
            w_accuracy = len(intersection) / len(self.valid_features)
            log_debug("W accuracy: ", w_accuracy)
            self.results_handler.add_result('loss', self.current_loss_value)
            self.results_handler.add_result('w_accuracy', w_accuracy)
            return w_accuracy


def show_heatmap(mat):
    sns.heatmap(mat, cmap='Blues')
    plt.show()


class PCAFeatureExtraction(FeatureExtractionAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w_optimizer = None
        self.n_components = kwargs['n_components']
        self.xi = kwargs['xi']
        self.sorted_indices = None
        self.use_loss_B = kwargs['use_loss_B']
        self.use_normalization = kwargs['use_normalization']
        self.norm = kwargs['norm']
        self.iterative = kwargs['iterative']
        self.epochs = kwargs['epochs']
        self.results_handler = kwargs['results_handler']
        self.should_accumulate_w = kwargs['accumulating_w'] if self.iterative else False
        self.easy_accumulation = kwargs['easy_accumulation'] if self.should_accumulate_w else False
        self.args = kwargs

    def get_relevant_features(self, X, amount=10):
        return self.sorted_indices[:amount]

    def train(self, X, y=None, metadata=None, epochs=1):
        log_debug('Starting training on PCAFeatureExtraction')
        # First stage - create all pairs of vectors

        # Second stage - For each pair - calculate compressibility ratio using PCA
        log_debug('Calculating PCA for matrix sized : ', X.shape)
        # TODO: Check what Lior said with not needing to subtract the mu
        pca = sklearn.decomposition.PCA(n_components=self.n_components)

        pca.fit(X)
        P = torch.from_numpy(pca.components_).float()  # Should be sized (n_components, n_features)
        log_debug('Finished calculating PCA')

        log_debug('Calculating B')
        B, C = self.calculate_B_C(P, X, y)
        log_debug('Finished calculating B, starting to calculate w')
        # Third stage - create w and Perform SGD on w where the loss
        # is -1 * mean(compressibility of batch)

        self.w_optimizer.set_vals(B, C, P)
        w = self.w_optimizer.optimize_w(epochs)

        # Fourth stage - save w
        self.sorted_indices = torch.argsort(w, descending=True)
        return w.data

    def train_wrapper(self, X, y=None, metadata=None):
        fake_B = torch.ones((1, 2, X.shape[1]))
        valid_features = metadata['n_relevant_features'] if 'n_relevant_features' in metadata else None
        if valid_features:
            valid_features = torch.arange(valid_features)
        self.w_optimizer = Optimizer(fake_B, fake_B, valid_features=valid_features, **self.args)
        if self.easy_accumulation:
            self.epochs = int(self.epochs / 10)
        if self.iterative:
            original_X = X.clone()
            if self.should_accumulate_w:
                accumulated_w = torch.ones((X.shape[1]))
            for i in range(self.epochs):
                if self.easy_accumulation:
                    w = self.train(X, y, metadata, epochs=100)
                else:
                    w = self.train(X, y, metadata, epochs=1)
                if self.should_accumulate_w:
                    accumulated_w *= w
                    accumulated_w /= torch.linalg.norm(accumulated_w, ord=self.norm)
                    X = original_X * accumulated_w
                    self.w_optimizer.print_status(accumulated_w)
                    self.w_optimizer.w.data = torch.ones((X.shape[1]))
                else:
                    X = w * original_X
                    self.w_optimizer.print_status(w)
                X = X - torch.mean(X, dim=0)
        else:
            self.train(X, y, metadata, epochs=self.epochs)

    def calculate_B_C(self, P, X, y):
        X = torch.unique(X, dim=0)
        indices_pairs = self.get_pairs(X, X.shape[0], y)
        left_indices = indices_pairs[:, 0]
        right_indices = indices_pairs[:, 1]
        log_debug('Starting to calculate all grades')
        u = X[left_indices]
        v = X[right_indices]
        grades = calculate_compression(u, v, P)
        groups_size = int(self.xi * indices_pairs.shape[0])
        B_grades, B_indices = torch.topk(grades, groups_size, largest=False)
        C_grades, C_indices = torch.topk(grades, groups_size, largest=True)

        log_debug('Finished calculating all grades')
        # Still second - Create B - the group of top xi pairs according to measure
        B = torch.stack((u[B_indices], v[B_indices]), dim=1)
        C = torch.stack((u[C_indices], v[C_indices]), dim=1)

        b_precent = torch.sum(y[left_indices[B_indices]] == y[right_indices[B_indices]]) / groups_size
        c_precent = torch.sum(y[left_indices[C_indices]] == y[right_indices[C_indices]]) / groups_size

        b_ratio = torch.sum(B_grades) / groups_size
        c_ratio = torch.sum(C_grades) / groups_size
        self.results_handler.add_result('b_precent', b_precent)
        self.results_handler.add_result('c_precent', c_precent)
        self.results_handler.add_result('b_ratio', b_ratio)
        self.results_handler.add_result('c_ratio', c_ratio)
        # log_info('Same group precentage in B is {}'.format(b_precent))
        log_info('Same group precentage in C is {}'.format(c_precent))

        return B, C

    def get_pairs(self, X, n, y):

        indices = torch.tensor([i for i in range(n)])
        log_debug('Using mashgrid to create all indices_pairs')
        left_indices, right_indices = torch.meshgrid(indices, indices)
        left_indices = left_indices.flatten()
        right_indices = right_indices.flatten()

        indices_pairs = torch.stack([left_indices, right_indices], dim=1)
        indices_pairs = indices_pairs[left_indices != right_indices]
        # indices_pairs = indices_pairs[torch.randperm(indices_pairs.shape[0])]

        log_debug('Finished. Some playing with indices.')

        return indices_pairs

    def __str__(self):
        return 'n={}, w_norm={}, loss_B={}'.format(self.n_components, self.use_normalization, self.use_loss_B)


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


def run_synthetic_experiment():
    n_components = 5
    times = 1
    seed = 42
    torch.manual_seed(seed)
    results = []
    names = ['L1', 'L2', 'No_norm']
    for i in range(times):
        X, y, name, metadata = synthetic_data_generator.get_synthetic_dataset(seed + i)
        first_algo = PCAFeatureExtraction(n_components=n_components, use_normalization=True, use_loss_B=False, norm=1)
        second_algo = PCAFeatureExtraction(n_components=n_components, use_normalization=True, use_loss_B=False, norm=2)
        third_algo = PCAFeatureExtraction(n_components=n_components, use_normalization=False, use_loss_B=False)
        # algos = [first_algo, second_algo, third_algo]
        # for algo, name in zip(algos, names):
        #     print('Testing vs metadata', metadata)
        #     result =algo.only_train(X, y, metadata)
        #     result['metadata'] = metadata
        #     result['name'] = name
        #     result['type'] = 'regular'
        #     results.append(result)
        #     print('Finished iteration {} with algo : {}'.format(i, name))
        #     print(results[i])
        #
        # first_algo = PCAFeatureExtraction(n_components=n_components, use_normalization=True, use_loss_B=False, norm=1)
        # second_algo = PCAFeatureExtraction(n_components=n_components, use_normalization=True, use_loss_B=False, norm=2)
        # third_algo = PCAFeatureExtraction(n_components=n_components, use_normalization=False, use_loss_B=False)
        # algos = [first_algo, second_algo, third_algo]

        for algo, name in zip(algos, names):
            print('Testing vs metadata', metadata)
            result = algo.iterative_train(X, y, metadata)
            result['metadata'] = metadata
            result['name'] = name
            result['type'] = 'iterative'
            results.append(result)
            print('Finished iteration {} with algo : {}'.format(i, name))
            print(results[i])
    print('Finished all iterations')
    print(results)
    # Saving results
    with open('.//sync//results3.pickle', 'wb') as f:
        import pickle
        pickle.dump(results, f)


if __name__ == '__main__':
    run_synthetic_experiment()
    exit()
    n_components = 5
    feature_amount = 100
    seed = 42
    torch.manual_seed(seed)

    algos = [Baseline()]
    # with_loss_B_with_normalization = PCAFeatureExtraction(n_components)
    # with_loss_B_with_normalization.use_loss_B = True
    # with_loss_B_with_normalization.use_normalization = True
    # without_loss_B_with_normalization = PCAFeatureExtraction(n_components)
    # without_loss_B_with_normalization.use_loss_B = False
    # without_loss_B_with_normalization.use_normalization = True
    # with_loss_B_without_normalization = PCAFeatureExtraction(n_components)
    # with_loss_B_without_normalization.use_loss_B = True
    # with_loss_B_without_normalization.use_normalization = False
    #
    # without_loss_B_without_normalization = PCAFeatureExtraction(n_components)
    # without_loss_B_without_normalization.use_loss_B = False
    # without_loss_B_without_normalization.use_normalization = False

    algo = PCAFeatureExtraction(n_components, use_normalization=True, use_loss_B=False)

    algos = [algo]

    with autograd.detect_anomaly():
        # Defining params
        log_debug('Starting to test algos : ', algos)
        for X, y, name, metadata in datasets.read_datasets():
            feature_jump = 10
            log_debug('Starting to test on dataset {}'.format(name))
            log_debug('X shape is {}'.format(X.shape))
            for algo in algos:
                log_debug('Starting to test algo : ', algo)
                algo.database_name = './/fake_b_c_datasets//{}.pt'.format(name)
                log_debug('Starting to test algo {} on dataset {}'.format(algo, name))
                algo.iterative_train(X, y, metadata)
            # amounts = [i for i in range(feature_jump, min(200, X.shape[-1]), feature_jump)]
            # results = []
            # for feature_amount in amounts:
            #     result = run_algo(algo, X, y, feature_amount=feature_amount)
            #     results.append(result)
            # results = torch.tensor(results)
        #     log_debug("\n\n==============")
        #     log_debug('Max result is {} in index {}'.format((torch.argmax(results) + 1) * 20, torch.max(results) * 100))
        #     log_debug("==============\n\n")
        #
        #     # Plotting results
        #     plt.plot(amounts, results, label=algo.__str__())
        # plt.legend()
        # plt.title('Dataset {}'.format(name))
        # plt.show()
