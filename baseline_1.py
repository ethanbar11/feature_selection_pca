import logging

import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import torch
import torch.optim
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

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
        self.feature_amount = kwargs['feature_amount']
        self.results_handler = kwargs['results_handler']

    # Expecting X to be numpy array of size (n_samples, n_features)
    def get_relevant_features(self, X):
        raise NotImplementedError

    def run_k_means(self, X, y):
        for n_features in self.feature_amount:
            NUM_OF_RUNS = 5
            features = self.get_relevant_features(X, n_features)
            n_clusters = len(torch.unique(y))
            X_used = X[:, features]
            accuracies = []
            seed = 42
            for i in range(NUM_OF_RUNS):
                kmeans = KMeans(n_clusters=n_clusters, random_state=seed + i).fit(X_used.cpu())
                y_pred = kmeans.labels_
                mutual_info_score = normalized_mutual_info_score(y, y_pred)
                accuracies.append(mutual_info_score)
            self.results_handler.add_result('NMI', (n_features, torch.mean(torch.tensor(accuracies))))

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


def get_pca(X, n_components, sacling=False, center=False, ):
    # TODO: Notice there is a difference of <eps = 1e-7 between my PCA and numpy's
    X = X.clone()
    if center:
        X = X - torch.mean(X, dim=0)
    if sacling:
        X = X / torch.std(X, dim=0)

    cov = torch.cov(X.T)
    eigenvalues, eigenvectors = torch.linalg.eig(cov)
    eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real
    eigenvalues, eigenvectors = eigenvalues[:n_components], eigenvectors[:, :n_components]
    return eigenvectors


class Optimizer:
    def __init__(self, B, C, **kwargs):
        self.use_loss_ratio = kwargs['use_loss_ratio'] if 'use_loss_ratio' in kwargs else False
        self.use_loss_C = kwargs['use_loss_C'] if 'use_loss_C' in kwargs else False
        self.use_loss_B = kwargs['use_loss_B'] if 'use_loss_B' in kwargs else False
        print('use_loss_C', self.use_loss_C)
        print('use_loss_B', self.use_loss_B)
        self.use_normalization = kwargs['use_normalization'] and not kwargs['accumulating_w']
        self.use_clamping = kwargs['use_clamping'] if 'use_clamping' in kwargs else True
        self.valid_features = kwargs['valid_features']
        self.BATCH_SIZE = int(5e4)
        self.LEARNING_RATE = kwargs['learning_rate']
        self.norm = kwargs['norm']
        self.results_handler = kwargs['results_handler']
        self.set_vals(B, C, torch.zeros((1, 1)))
        log_debug("Starting to train w. Number of batches: ", len(self.B_batches))
        self.w = torch.ones((B.shape[-1]), requires_grad=True, device=kwargs['device'])
        self.current_loss_value = None
        self.optimizer = torch.optim.SGD(params=[self.w], lr=self.LEARNING_RATE)

    # You have to call for set_vals before using.
    def set_vals(self, B, C, P):
        self.B = B
        self.C = C
        self.P = P
        # self.P = torch.abs(P)
        self.B_batches = torch.split(B, self.BATCH_SIZE, dim=0)
        self.C_batches = torch.split(C, self.BATCH_SIZE, dim=0)

    def optimize_w_one_epoch(self, epochs):
        # B is a matrix of size (chi * n**2 , d)
        # w is a vector of size (d, 1)
        # P is a matrix of size (k, d)
        # Split B into batches of rows

        for batch_b, batch_c in zip(self.B_batches, self.C_batches):
            # Calculating the gradient of the loss function
            # with respect to w
            self.optimizer.zero_grad()

            u = batch_b[:, 0, :]
            v = batch_b[:, 1, :]
            u2 = batch_c[:, 0, :]
            v2 = batch_c[:, 1, :]
            loss = torch.tensor(0, dtype=torch.float32, device=self.w.device)
            if self.use_loss_C:
                loss_C = torch.mean(calculate_compression_w(u2, v2, self.P, self.w)) * (-1.0)
                loss += loss_C
            if self.use_loss_B:
                loss_B = torch.mean(calculate_compression_w(u, v, self.P, self.w)) * (-1.0)
                loss += loss_B
            if self.use_loss_ratio:
                loss_B = torch.mean(calculate_compression_w(u, v, self.P, self.w))
                loss_C = torch.mean(calculate_compression_w(u2, v2, self.P, self.w))
                loss_ratio = (loss_C / loss_B) * (-1.0)
                loss += loss_ratio

            loss.backward()
            self.current_loss_value = loss.item()
            self.optimizer.step()
            if self.use_normalization:
                self.w.data = self.w.data / torch.linalg.norm(self.w.data, ord=self.norm)
            if self.use_clamping:
                self.w.data = torch.clamp(self.w.data, min=0.0, max=1.0)
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


class PCAStatistics(FeatureExtractionAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_components = kwargs['n_components']
        self.results_handler = kwargs['results_handler']
        self.feature_amount = kwargs['feature_amount'] if 'feature_amount' in kwargs else None
        self.order = None
        # Only for running on real data

    def train_wrapper(self, X, y=None, metadata=None):
        P = get_pca(X, self.n_components).T * (-1.0)
        vals = torch.mean(torch.abs(P), dim=0)
        self.order = torch.argsort(vals, descending=True)


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
        self.should_accumulate_w = kwargs['accumulating_w'] if self.iterative else False
        self.easy_accumulation = kwargs['easy_accumulation'] if self.should_accumulate_w else False
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
        self.pca_only_on_true_featuers = kwargs[
            'pca_only_on_true_features'] if 'pca_only_on_true_features' in kwargs else False
        self.test_on_k_means = kwargs['test_on_k_means'] if 'test_on_k_means' in kwargs else False
        self.fake_groups = kwargs['use_fake_groups'] if 'use_fake_groups' in kwargs else False
        self.args = kwargs
        # Only for running on real data

    def get_relevant_features(self, X, amount=10):
        return self.sorted_indices[:amount]

    def train(self, X, y=None, metadata=None, epochs=1, direct_train=False):
        if not direct_train:
            log_debug('Starting training on PCAFeatureExtraction')
            # First stage - create all pairs of vectors

            # Second stage - For each pair - calculate compressibility ratio using PCA
            log_debug('Calculating PCA for matrix sized : ', X.shape)
            if self.pca_only_on_true_featuers and metadata is not None:
                X_orig = X.clone()
                n_relevant_features = metadata['n_relevant_features'] if 'n_relevant_features' in metadata else None
                non_valid = X.shape[-1] - n_relevant_features
                mask = torch.cat((torch.ones((X.shape[0], n_relevant_features), device=self.device),
                                  torch.zeros((X.shape[0], non_valid), device=self.device)), dim=1)
                X = X * mask
                P = get_pca(X, self.n_components).T * (-1.0)
                X = X_orig
            else:
                P = get_pca(X, self.n_components).T * (-1.0)
            log_debug('Finished calculating PCA')

            log_debug('Calculating B')
            B, C = self.calculate_B_C(P, X, y)
            log_debug('Finished calculating B, starting to calculate w')
            # Third stage - create w and Perform SGD on w where the loss
            # is -1 * mean(compressibility of batch)

            self.w_optimizer.set_vals(B, C, P)

        P = get_pca(X, self.n_components).T * (-1.0)
        self.w_optimizer.P = P
        w = self.w_optimizer.optimize_w_one_epoch(epochs)

        # Fourth stage - save w
        self.sorted_indices = torch.argsort(w, descending=True)
        return w.data

    def train_wrapper(self, X, y=None, metadata=None):
        fake_B = torch.ones((1, 2, X.shape[1]))
        valid_features = metadata['n_relevant_features'] if 'n_relevant_features' in metadata else None
        if valid_features:
            valid_features = torch.arange(valid_features, device=self.device)
        self.w_optimizer = Optimizer(fake_B, fake_B, valid_features=valid_features, **self.args)
        if self.easy_accumulation:
            self.epochs = int(self.epochs / 10)
        if self.iterative:
            original_X = X.clone()
            if self.should_accumulate_w:
                accumulated_w = torch.ones((X.shape[1]), device=self.device)
            for i in range(self.epochs):
                if self.easy_accumulation:
                    w = self.train(X, y, metadata, epochs=100)
                else:
                    w = self.train(X, y, metadata, epochs=1,direct_train=True)
                if self.should_accumulate_w:
                    accumulated_w *= w
                    accumulated_w /= torch.linalg.norm(accumulated_w, ord=self.norm)
                    X = original_X * accumulated_w
                    self.w_optimizer.print_status(accumulated_w)
                    self.w_optimizer.w.data = torch.ones((X.shape[1]), device=self.device)
                else:
                    X = w * original_X
                    print('Got here')
                    self.w_optimizer.print_status(w)
                X = X - torch.mean(X, dim=0)
                if i % 10 == 0:
                    self.run_k_means(X, y)

        else:
            for epoch in range(self.epochs):
                if epoch == 0:
                    w = self.train(X, y, metadata, epochs=1, direct_train=False)
                else:
                    w = self.train(X, y, metadata, epochs=1, direct_train=True)
                self.w_optimizer.print_status(w)
                if epoch % 10 == 0:
                    self.run_k_means(X, y)

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
        if self.fake_groups:
            # Getting "pure" B,C
            y_left = y[left_indices]
            y_right = y[right_indices]
            indices_by_clusters = torch.eq(y_left, y_right).float()
            B_grades, B_indices = torch.topk(indices_by_clusters, groups_size, largest=True)
            C_grades, C_indices = torch.topk(indices_by_clusters, groups_size, largest=False)
        else:
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
        log_info('Same group precentage in B is {}'.format(b_precent))
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
