import pickle

import torch
from scipy.optimize import linear_sum_assignment
import numpy as np

# def map(y, y_pred):
#     """
#     :param y: ground truth
#     :param y_pred: predicted labels
#     :return: the accuracy
#     """
#     y = y.cpu().numpy()
#     y_pred = y_pred.cpu().numpy()
#     n = len(y)
#     cost_matrix = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             cost_matrix[i, j] = 1 if y[i] == y[j] and y_pred[i] == y_pred[j] else 0
#     row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
#     return np.sum(cost_matrix[row_ind, col_ind]) / y.shape[0]
from results_handler import RealResultsHandler

if __name__ == '__main__':
    path = 'results//results.pickle'
    with open(path, 'rb') as f:
        exps = pickle.load(f)
        results = RealResultsHandler('gpu')
        results.past_experiments_tot_data = exps
        results.plot_experiments_by_feature_amount()