import logging
import seaborn as sns
import matplotlib.pyplot as plt

import torch


def show_heatmap(mat):
    sns.heatmap(mat, cmap='Blues')
    plt.show()


def log_info(*args):
    logging.info(args_to_string(args))


def log_debug(*args):
    logging.debug(args_to_string(args))


# Convert args to one string
def args_to_string(args):
    return ' '.join([str(arg) for arg in args])


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
