import logging

import torch
import torch.optim

from algorithms import FeatureExtractionAlgorithm
from misc import log_debug, get_pca, log_info

logging.basicConfig(level=logging.INFO, format='%(message)s')


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
        self.use_loss_ratio = kwargs['use_loss_ratio'] if 'use_loss_ratio' in kwargs else False
        self.use_loss_C = kwargs['use_loss_C'] if 'use_loss_C' in kwargs else False
        self.use_loss_B = kwargs['use_loss_B'] if 'use_loss_B' in kwargs else False
        self.use_normalization = kwargs['use_normalization'] and not kwargs['accumulating_w']
        self.use_clamping = kwargs['use_clamping'] if 'use_clamping' in kwargs else True
        # self.valid_features = kwargs['valid_features']
        self.BATCH_SIZE = int(5e3)
        self.LEARNING_RATE = kwargs['learning_rate']
        self.norm = kwargs['norm']
        self.results_handler = kwargs['results_handler']
        self.set_vals(B, C, torch.zeros((1, 1)))
        log_debug("Starting to train w. Number of batches: ", len(self.B_batches))
        self.device = kwargs['device']
        self.w = torch.ones((B.shape[-1]), requires_grad=True, device=self.device)
        self.optimizer = torch.optim.SGD(params=[self.w], lr=self.LEARNING_RATE)
        self.current_epoch = 0

    # You have to call for set_vals before using.
    def set_vals(self, B, C, P):
        self.B = B
        self.C = C
        self.P = P
        # self.P = torch.abs(P)
        self.B_batches = torch.split(B, self.BATCH_SIZE, dim=0)
        self.C_batches = torch.split(C, self.BATCH_SIZE, dim=0)

    def optimize_w_one_epoch(self):
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
            loss_B = calculate_compression_w(u, v, self.P, self.w)
            loss_C = calculate_compression_w(u2, v2, self.P, self.w)
            if self.use_loss_C:
                loss += torch.mean(loss_C) * (-1.0)
            if self.use_loss_B:
                loss += torch.mean(loss_B)
            if self.use_loss_ratio:  # and self.current_epoch > 10:
                # loss_ratio = (loss_B / loss_C)
                # loss_before_hinge = torch.cat(
                #     ((loss_C - loss_B + m).unsqueeze(-1), torch.zeros(loss_B.shape[0], device=self.device)
                #      .unsqueeze(-1)), dim=1)
                # loss_ratio, _ = torch.max(loss_before_hinge, dim=1)
                loss_ratio = torch.mean(loss_ratio.float())
                loss += loss_ratio

            loss.backward()
            self.optimizer.step()
            if self.use_normalization:
                self.w.data = self.w.data / torch.linalg.norm(self.w.data, ord=self.norm)
            if self.use_clamping:
                self.w.data = torch.clamp(self.w.data, min=0.0, max=1.0)
        self.current_epoch += 1
        return self.w.detach()


class ClassicGroup:
    def __init__(self, X, y=None, metadata=None, **kwargs):
        self.xi = kwargs['xi']
        self.X = X
        self.y = y
        self.metadata = metadata
        self.results_handler = kwargs['results_handler']
        self.n_components = kwargs['n_components']
        self.pca_only_on_true_featuers = kwargs['pca_only_on_true_features']
        self.device = kwargs['device']

    def update_X(self, X):
        self.X = X

    def get_pairs(self, X, n, y):
        indices = torch.tensor([i for i in range(n)])
        log_debug('Using mashgrid to create all indices_pairs')
        left_indices, right_indices = torch.meshgrid(indices, indices)
        left_indices = left_indices.flatten()
        right_indices = right_indices.flatten()

        indices_pairs = torch.stack([left_indices, right_indices], dim=1)
        indices_pairs = indices_pairs[left_indices != right_indices]

        log_debug('Finished. Some playing with indices.')

        return indices_pairs

    def calculate_P(self):
        X = self.X
        if self.pca_only_on_true_featuers and self.metadata is not None:
            n_relevant_features = self.metadata[
                'n_relevant_features'] if 'n_relevant_features' in self.metadata else None
            non_valid = X.shape[-1] - n_relevant_features
            mask = torch.cat((torch.ones((X.shape[0], n_relevant_features), device=self.device),
                              torch.zeros((X.shape[0], non_valid), device=self.device)), dim=1)
            X = X * mask
        return get_pca(X, self.n_components).T * (-1.0)

    def calculate_B_C_P(self):
        X = torch.unique(self.X, dim=0)
        indices_pairs = self.get_pairs(X, X.shape[0], self.y)
        left_indices = indices_pairs[:, 0]
        right_indices = indices_pairs[:, 1]
        log_debug('Starting to calculate all grades')
        u = X[left_indices]
        v = X[right_indices]
        P = self.calculate_P()
        grades = calculate_compression(u, v, P)
        groups_size = int(self.xi * indices_pairs.shape[0])
        B_grades, B_indices, C_grades, C_indices = self.get_B_C_grades_and_indices(grades, groups_size, left_indices,
                                                                                   right_indices)

        log_debug('Finished calculating all grades')
        # Still second - Create B - the group of top xi pairs according to measure
        B = torch.stack((u[B_indices], v[B_indices]), dim=1)
        C = torch.stack((u[C_indices], v[C_indices]), dim=1)

        b_precent = torch.sum(self.y[left_indices[B_indices]] == self.y[right_indices[B_indices]]) / groups_size
        c_precent = torch.sum(self.y[left_indices[C_indices]] == self.y[right_indices[C_indices]]) / groups_size

        b_ratio = torch.sum(B_grades) / groups_size
        c_ratio = torch.sum(C_grades) / groups_size
        self.results_handler.add_result('b_precent', b_precent)
        self.results_handler.add_result('c_precent', c_precent)
        self.results_handler.add_result('b_ratio', b_ratio)
        self.results_handler.add_result('c_ratio', c_ratio)

        return B, C, P

    def get_B_C_grades_and_indices(self, grades, groups_size, left_indices, right_indices):
        B_grades, B_indices = torch.topk(grades, groups_size, largest=False)
        C_grades, C_indices = torch.topk(grades, groups_size, largest=True)
        return B_grades, B_indices, C_grades, C_indices


class FakeBAndC(ClassicGroup):
    def get_B_C_grades_and_indices(self, grades, groups_size, left_indices, right_indices):
        y_left = self.y[left_indices]
        y_right = self.y[right_indices]
        indices_by_clusters = torch.eq(y_left, y_right).float()
        B_grades, B_indices = torch.topk(indices_by_clusters, groups_size, largest=True)
        C_grades, C_indices = torch.topk(indices_by_clusters, groups_size, largest=False)
        return B_grades, B_indices, C_grades, C_indices


class RatioGroup(ClassicGroup):
    def __init__(self, X, y=None, metadata=None, **kwargs):
        super().__init__(X, y, metadata, **kwargs)

    def get_B_C_grades_and_indices(self, grades, groups_size, left_indices, right_indices):
        # TODO: Maybe add this as D and E groups.
        y_left = self.y[left_indices]
        y_right = self.y[right_indices]
        indices_by_clusters = torch.eq(y_left, y_right).to(self.device)
        grades_equal = grades * indices_by_clusters
        grades_equal[grades_equal == 0] = 2  # Above max threshold
        grades_not_equal = grades * (~indices_by_clusters)
        B_grades, B_indices = torch.topk(grades_equal, groups_size, largest=False)
        C_grades, C_indices = torch.topk(grades_not_equal, groups_size, largest=True)
        return B_grades, B_indices, C_grades, C_indices


class PCAFeatureExtraction(FeatureExtractionAlgorithm):
    def __init__(self, X, y=None, metadata=None, **kwargs):
        super().__init__(**kwargs)
        self.calculate_P_each_time = kwargs['update_P'] if 'update_P' in kwargs else False
        self.w_optimizer = None
        self.n_components = kwargs['n_components']
        self.sorted_indices = None
        self.use_loss_B = kwargs['use_loss_B']
        self.use_normalization = kwargs['use_normalization']
        self.norm = kwargs['norm']
        self.epochs = kwargs['epochs']
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
        self.pca_only_on_true_featuers = kwargs[
            'pca_only_on_true_features'] if 'pca_only_on_true_features' in kwargs else False
        self.fake_groups = kwargs['use_fake_groups'] if 'use_fake_groups' in kwargs else False
        self.args = kwargs
        self.X = X
        self.y = y
        self.metadata = metadata
        if self.fake_groups and not kwargs['use_loss_ratio']:
            self.group_manager = FakeBAndC(X, y, metadata, **self.args)
        elif kwargs['use_loss_ratio']:
            self.group_manager = RatioGroup(X, y, metadata, **self.args)
        else:
            self.group_manager = ClassicGroup(X, y, metadata, **self.args)
        fake_B = torch.ones((1, 2, self.X.shape[1]))
        # if self.metadata:
        # valid_features = self.metadata['n_relevant_features'] if 'n_relevant_features' in metadata else None
        # if valid_features:
        #     valid_features = torch.arange(valid_features, device=self.device)
        self.w_optimizer = Optimizer(fake_B, fake_B, **self.args)

    def get_relevant_features(self, X, amount=10):
        return self.sorted_indices[:amount]

    def train(self):
        B, C, P = self.group_manager.calculate_B_C_P()
        self.w_optimizer.set_vals(B, C, P)
        w = torch.ones(self.X.shape[-1], device=self.device)
        for epoch in range(self.epochs):
            if self.calculate_P_each_time and epoch %10 == 0:
                self.group_manager.update_X(self.X * w)
                P = self.group_manager.calculate_P()
                self.w_optimizer.set_vals(B, C, P)
            w = self.w_optimizer.optimize_w_one_epoch()
            if epoch % 2 == 0:
                self.results_handler.add_result('w', w)
        self.sorted_indices = torch.argsort(w, descending=True)

    def __str__(self):
        return 'n={}, w_norm={}, loss_B={}'.format(self.n_components, self.use_normalization, self.use_loss_B)


class IterativePCAFeatureExtraction(PCAFeatureExtraction):
    def __init__(self, X, y=None, metadata=None, **kwargs):
        super().__init__(X, y, metadata, **kwargs)
        self.should_accumulate_w = kwargs['accumulating_w']

    def train(self):
        X = self.X
        if self.should_accumulate_w:
            accumulated_w = torch.ones((self.X.shape[1]), device=self.device)
        for i in range(self.epochs):
            self.group_manager.update_X(X)
            B, C, P = self.group_manager.calculate_B_C_P()
            self.w_optimizer.set_vals(B, C, P)
            w = self.w_optimizer.optimize_w_one_epoch()
            if self.should_accumulate_w:
                accumulated_w *= w
                accumulated_w /= torch.linalg.norm(accumulated_w, ord=self.norm)
                X = self.X * accumulated_w
                self.w_optimizer.w.data = torch.ones((X.shape[1]), device=self.device).data
            else:
                X = w * self.X
            X = X - torch.mean(X, dim=0)
            self.results_handler.add_result('w', w)

    def __str__(self):
        return 'Iterative n={}, w_norm={}, loss_B={}, '.format(self.n_components, self.use_normalization,
                                                               self.use_loss_B)
