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
    def __init__(self, B, C, P, use_loss_B=False, use_normalization=False, use_clamping=True, valid_features=None):
        self.use_loss_B = use_loss_B
        self.use_normalization = use_normalization
        self.use_clamping = use_clamping
        self.valid_features = valid_features
        self.BATCH_SIZE = int(1e4)
        self.LEARNING_RATE = 0.05
        self.EPOCHS = 1
        self.P = P
        self.B = B
        self.C = C
        self.B_batches = torch.split(B, self.BATCH_SIZE, dim=0)
        self.C_batches = torch.split(C, self.BATCH_SIZE, dim=0)
        self.set_vals(B, C, P)
        log_debug("Starting to train w. Number of batches: ", len(self.B_batches))
        self.w = torch.ones((B.shape[-1]), requires_grad=True)
        self.current_loss_value = None
        self.optimizer = torch.optim.SGD(params=[self.w], lr=self.LEARNING_RATE)

    def set_vals(self, B, C, P):
        # self.B = B
        # self.C = C
        # self.P = P
        self.B_batches = torch.split(B, self.BATCH_SIZE, dim=0)
        self.C_batches = torch.split(C, self.BATCH_SIZE, dim=0)

    def optimize_w(self):
        # B is a matrix of size (chi * n**2 , d)
        # w is a vector of size (d, 1)
        # P is a matrix of size (k, d)
        # Split B into batches of rows
        for epoch in range(self.EPOCHS):

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
                    self.w.data = self.w.data / torch.linalg.norm(self.w.data, ord=1)
                    # Softmax normalization
                    # self.w.data = torch.nn.functional.softmax(self.w.data, dim=0)
                if self.use_clamping:
                    self.w.data = torch.clamp(self.w.data, min=0.0, max=1.0)
        return self.w.detach()

    def print_status(self, epoch):
        log_info("Epoch: ", epoch, " Loss: ", self.current_loss_value)
        if self.valid_features is not None:
            biggest_weight_indices = torch.argsort(self.w, descending=True)[:len(self.valid_features)]
            combined = torch.cat((biggest_weight_indices, self.valid_features))
            uniques, counts = combined.unique(return_counts=True)
            intersection = uniques[counts > 1]
            w_accuracy = len(intersection) / len(self.valid_features)
            log_info("W accuracy: ", w_accuracy)
            # log_debug('Weights : ', self.w)


class PCAFeatureExtraction(FeatureExtractionAlgorithm):
    def __init__(self, n_components, fake_groups=False, database_name=None, use_loss_B=False,
                 use_normalization=False, **kwargs):
        super().__init__(**kwargs)
        self.w_optimizer = None
        self.n_components = n_components
        self.xi = 0.01  # between [0,1], size of B out of n^2 pairs
        self.sorted_indices = None
        self.fake_groups = fake_groups
        self.database_name = database_name
        self.use_loss_B = use_loss_B
        self.use_normalization = use_normalization

    def get_relevant_features(self, X, amount=10):
        return self.sorted_indices[:amount]

    def calculate_optimal_n_components(self, X, y):
        return self.n_components
        table = []
        for n_components in range(1, 2):
            pca = sklearn.decomposition.PCA(n_components=n_components)

            pca.fit(X)
            P = torch.from_numpy(pca.components_).float()  # Should be sized (n_components, n_features)
            log_debug('n_components: ', n_components)
            b_precent, c_precent, b_mean, c_mean = self.calculate_B_C(P, X, y)
            table.append([n_components, b_precent, c_precent, b_mean, c_mean])
        table = np.array(table)
        log_debug('Saving table')
        log_debug(table)
        np.savetxt('table.csv', table, delimiter=',')
        exit()

    def train(self, X, y=None, metadata=None):
        self.calculate_optimal_n_components(X, y)
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
        if not self.fake_groups:
            B, C = self.calculate_B_C(P, X, y)
        else:
            B, C = self.calculate_B_C_fake_groups(X, y)
        log_debug('Finished calculating B, starting to calculate w')
        # Third stage - create w and Perform SGD on w where the loss
        # is -1 * mean(compressibility of batch)

        valid_features = None
        if metadata:
            valid_features = torch.arange(metadata['n_relevant_features'])
        if not self.w_optimizer:
            self.w_optimizer = Optimizer(B, C, P, use_loss_B=self.use_loss_B, use_normalization=self.use_normalization,
                                         valid_features=valid_features)
        self.w_optimizer.set_vals(B, C, P)
        w = self.w_optimizer.optimize_w()

        # Fourth stage - save w
        self.sorted_indices = torch.argsort(w, descending=True)
        return w.data

    def iterative_train(self, X, y=None, metadata=None):
        # Calculates C each time and updates w
        EPOCHS = 100
        # First train
        self.train(X, y, metadata)
        self.w_optimizer.EPOCHS = 1
        original_X = X.clone()
        for epoch in range(EPOCHS - 1):
            w = self.train(X, y, metadata)
            # Perform softmax on weights
            X = w * original_X

            # print(w)
            self.w_optimizer.print_status(epoch)

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
        log_info('Same group precentage in B is {}'.format(b_precent))
        log_info('Same group precentage in C is {}'.format(c_precent))

        log_debug('Average grade in B is {}'.format(torch.mean(B_grades)))
        b_mean = torch.mean(B_grades)
        log_debug('Average grade in C is {}'.format(torch.mean(C_grades)))
        c_mean = torch.mean(C_grades)
        return B, C

    def calculate_B_C_fake_groups(self, X, y):
        # Check if database file exists
        if self.database_name and os.path.exists(self.database_name):
            log_debug('Loading database from file')
            B, C = torch.load(self.database_name)
            return B, C

        else:
            log_debug('Creating fake groups by myself...')
            n = X.shape[0]
            pairs, left_y, right_y = self.get_pairs(X, X.shape[0], y)
            # Validates I'm not taking from the same cluster

            pairs = pairs[torch.randperm(pairs.shape[0])]
            size = int(self.xi * n ** 2)
            B = []
            C = []
            for i in range(n ** 2):
                if i % 100000 == 0:
                    log_debug(
                        'Finished {} pairs, B is {}%, C is {}%'.format(i, len(B) * 100 / size, len(C) * 100 / size))
                same_class = left_y[i] == right_y[i]
                equal_vectors = torch.equal(pairs[i][0], pairs[i][1])
                if not equal_vectors:
                    if same_class and len(B) < size:
                        B.append(pairs[i])
                    elif not same_class and len(C) < size:
                        C.append(pairs[i])
                elif len(B) == size and len(C) == size:
                    break
            B = torch.stack(B, dim=0)
            C = torch.stack(C, dim=0)
            if self.database_name:
                log_debug('Saving database to file', self.database_name)
                torch.save((B, C), self.database_name)
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


if __name__ == '__main__':
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

    algo = PCAFeatureExtraction(n_components,use_normalization=True,use_loss_B=False)

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
