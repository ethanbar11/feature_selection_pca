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


def get_gd_pca(X, current_P):
    X = X.clone()
    k = current_P.shape[-1]
    LEARNING_RATE = 0.1

    real_P = get_pca(X.T, k)
    cov = torch.cov(X)
    # lam = torch.rand((k, k), requires_grad=True)
    # lam = torch.rand((k, k), requires_grad=True)
    lam = torch.rand(1, requires_grad=True)  # torch.rand((k, k), requires_grad=True)
    optimizer = torch.optim.SGD([current_P, lam], lr=LEARNING_RATE, maximize=True)
    for epoch in range(200):
        optimizer.zero_grad()
        left_side_loss = torch.matmul(current_P.T, torch.matmul(cov, current_P))
        right_side_loss = torch.matmul((lam * torch.eye(k)), (torch.matmul(current_P.T, current_P) - 1))
        loss = torch.sum(left_side_loss - right_side_loss)
        loss.backward()
        optimizer.step()
        # current_P.data = current_P.data / torch.norm(current_P.data, dim=0)
    print('solved solution', real_P)
    print('Numpy solution', get_sklearn_pca(X.T, k))
    print('current_P', current_P)


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
    eigenvectors = eigenvectors[:, torch.argsort(eigenvalues, descending=True)]
    eigenvalues = eigenvalues[torch.argsort(eigenvalues, descending=True)]
    eigenvalues, eigenvectors = eigenvalues[:n_components], eigenvectors[:, :n_components]
    return eigenvectors


def get_sklearn_pca(X, n_components, sacling=False, center=False, ):
    import numpy as np
    from sklearn.decomposition import PCA
    X = X.numpy()
    pca = PCA(n_components=n_components)
    pca.fit(X)
    return torch.from_numpy(pca.components_)


if __name__ == '__main__':
    n = 10
    d = 5
    k = 2
    seed = 42
    torch.manual_seed(seed)
    X = torch.randint(0, 5, (d, n)).float()
    X = X - torch.mean(X, dim=0)
    X = X / torch.std(X, dim=0)

    current_P = torch.rand((d, k), requires_grad=True)

    print(get_gd_pca(X, current_P))