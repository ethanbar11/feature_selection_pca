import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import synthetic_data_generator
import torch
import torch.optim
import sklearn.cluster
from sklearn.preprocessing import normalize
from sklearn.metrics.cluster import normalized_mutual_info_score


class FeatureExtractionAlgorithm:
    # Expecting X to be numpy array of size (n_samples, n_features)
    def get_relevant_features(self, X):
        raise NotImplementedError


class Baseline(FeatureExtractionAlgorithm):
    def get_relevant_features(self, X,amount=0):

        return [i for i in range(amount)]


# This is specific to the synthetic data generator
class Perfect(FeatureExtractionAlgorithm):
    def get_relevant_features(self, X):
        return [i for i in range(X.shape[1]) if i < 5]


def calculate_compression(u, v, P):
    diff = u - v
    outcome = torch.linalg.norm(diff) / torch.linalg.norm(torch.matmul(diff, P.T))
    return outcome


def calculate_compression_w(u, v, P, w):
    weighted_diff = w * (u - v)
    weighted_diff_P = torch.matmul(weighted_diff, P.T)
    outcome = torch.linalg.norm(weighted_diff_P, dim=1) / torch.linalg.norm(weighted_diff, dim=1)
    num = torch.linalg.norm(weighted_diff, dim=1).mean()
    dom = torch.linalg.norm(weighted_diff_P, dim=1).mean()
    # print('Numerator : ', num)
    # print('Dominator : ', dom)
    return outcome


# def calculate_compression_diff(u, v, P, w):
#     diff = u - v
#     diff_w = w * diff
#     outcome = torch.linalg.norm(diff) / torch.linalg.norm(torch.matmul(P, diff))
#     return outcome


def optimize_w(X, B, C, P, w=None, metadata=None, pca=None):
    # B is a matrix of size (chi * n**2 , d)
    # w is a vector of size (d, 1)
    # P is a matrix of size (k, d)
    # Split B into batches of rows

    BATCH_SIZE = B.shape[0]
    LEARNING_RATE = 0.1
    EPOCHS = 10000
    n = B.shape[0] // BATCH_SIZE
    B_batches = torch.split(B, BATCH_SIZE, dim=0)
    if w is None:
        w = torch.ones((B.shape[-1]), requires_grad=True)
    loss = None
    optimizer = torch.optim.SGD(params=[w], lr=LEARNING_RATE, weight_decay=0.1)
    for epoch in range(EPOCHS):

        # pca.fit(X * w.detach().numpy())
        # P = torch.from_numpy(pca.components_)  # Should be sized (n_components, n_features)

        for batch in B_batches:
            # Calculating the gradient of the loss function
            # with respect to w
            optimizer.zero_grad()

            u = batch[:, 0, :]
            v = batch[:, 1, :]
            u2 = C[:, 0, :]
            v2 = C[:, 1, :]
            # loss_B = torch.mean(calculate_compression_w(u, v, P, w))
            loss_C = torch.mean(calculate_compression_w(u2, v2, P, w)) * (-1.0)
            # loss = loss_B + loss_C
            loss_C.backward()
            optimizer.step()
    return w.detach()


def show_heatmap(mat, name=None):
    sns.set_theme()
    mat = np.abs(mat)
    if name:
        np.save(name, mat)

    ax = sns.heatmap(mat)
    print(mat)
    plt.show()


def pca_my_implementation(X, k):
    # X is a matrix of size (n, d)
    # k is the number of components to keep
    d = X.shape[1]
    mask = torch.ones(d, d) - torch.eye(d)
    cov = torch.cov(X.T) * mask
    show_heatmap(cov)
    eigen_values, eigen_vectors = np.linalg.eig(cov)
    # sort the eigenvalues in descending order
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    # similarly sort the eigenvectors
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    eigenvector_subset = sorted_eigenvectors[:, :k]
    return torch.from_numpy(eigenvector_subset)


class PCAFeatureExtraction(FeatureExtractionAlgorithm):
    def __init__(self, n_components):
        self.n_components = n_components
        self.xi = 0.01  # between [0,1], size of B out of n^2 pairs
        self.sorted_indices = None

    def get_relevant_features(self, X, amount=10):
        return self.sorted_indices[:amount]
        # pca = sklearn.decomposition.PCA(n_components=self.n_components)
        # pca.fit(X)
        # return pca.components_

    def train(self, X, y=None, metadata=None):
        print('Starting training on PCAFeatureExtraction')
        # First stage - create all pairs of vectors
        n = X.shape[0]
        pairs, left_y, right_y = self.get_pairs(X, n, y)

        print('Created all pairs')
        # Second stage - For each pair - calculate compressibility ratio using PCA
        print('Calculating PCA for matrix sized : ', X.shape)
        # TODO: Check what Lior said with not needing to substract the mu
        pca = sklearn.decomposition.PCA(n_components=self.n_components)

        pca.fit(X)
        P = torch.from_numpy(pca.components_).float()  # Should be sized (n_components, n_features)
        print('Finished calculating PCA')

        print('Calculating B')
        B, C = self.calculate_B_C(P, left_y, n, pairs, right_y)
        print('Finished calculating B, starting to calculate w')
        # Third stage - create w and Perform SGD on w where the loss
        # is -1 * mean(compressibility of batch)

        w = optimize_w(X, B, C, P, None, metadata, pca)

        # Fourth stage - save w
        self.sorted_indices = torch.argsort(w, descending=True)

    def calculate_B_C(self, P, left_y, n, pairs, right_y):
        grades = []
        # TODO: Perform this whole operation together with removing duplicates
        # pairs before, and then just calculate the grades.
        print('Starting to calculate all grades')
        for i in range(n ** 2):
            if not torch.equal(pairs[i][0], pairs[i][1]):
                grade = calculate_compression(pairs[i][0], pairs[i][1], P)
                same_class = left_y[i] == right_y[i]
                grades.append((grade, same_class, pairs[i]))
        # Still second - Create B - the group of top xi pairs according to measure
        # TODO: Change back to x[0], x[1] only for debugging
        print('Sorting grades')
        grades = sorted(grades, key=lambda x: x[1], reverse=True)

        B = grades[:int(self.xi * n ** 2)]
        C = grades[-int(self.xi * n ** 2):]

        print('Same group precentage in B is {}'.format(sum([x[1] for x in B]) / len(B)))
        print('Same group precentage in C is {}'.format(sum([x[1] for x in C]) / len(C)))
        print('Same group precentage in all pairs is {}'.format(sum([x[1] for x in grades]) / len(grades)))
        B = torch.stack([x[2] for x in B], dim=0)
        C = torch.stack([x[2] for x in C], dim=0)
        return B, C

    def get_pairs(self, X, n, y):
        indices = torch.tensor([i for i in range(n)])
        left_indices, right_indices = torch.meshgrid(indices, indices)
        left_indices = left_indices.flatten()
        right_indices = right_indices.flatten()
        left = torch.index_select(X, 0, left_indices)
        right = torch.index_select(X, 0, right_indices)
        left_y = torch.index_select(y, 0, left_indices)
        right_y = torch.index_select(y, 0, right_indices)
        pairs = torch.stack((left, right), dim=1)
        return pairs, left_y, right_y


# Running k-means and returning accuracy based on algo.
def run_algo(algo, X, y, seed=0, feature_amount=None):
    NUM_OF_RUNS = 1
    features = algo.get_relevant_features(X, feature_amount)
    print("Features used: ", features)
    n_clusters = len(torch.unique(y))
    X_used = X[:, features]
    accuracies = []
    for i in range(NUM_OF_RUNS):
        kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=seed + i).fit(X_used)
        y_pred = kmeans.labels_
        mutual_info_score = normalized_mutual_info_score(y, y_pred)
        accuracies.append(mutual_info_score)
        print("{} Mutual info score:".format(i), mutual_info_score)
    print("Average accuracy: ", torch.mean(torch.tensor(accuracies)))


if __name__ == '__main__':
    import h5py

    path = 'usps.h5'
    X = None
    y = None
    with h5py.File(path, 'r') as hf:
        train = hf.get('train')
        X = torch.from_numpy(train.get('data')[:]).float()
        y = torch.from_numpy(train.get('target')[:]).float()
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]
    n_components = 30
    feature_amount = 100
    print('Creating feature selection'
           '-')
    algo = Baseline()
    # algo.train(X, y)
    run_algo(algo, X, y,feature_amount=feature_amount)

# if __name__ == '__main__':
#     X, y, metadata = synthetic_data_generator.main()
#
#     X = torch.from_numpy(X)
#     y = torch.from_numpy(y)
#
#     k = 2
#     print("PCA k:", k)
#     algo = PCAFeatureExtraction(k)
#     algo.train(X, y, metadata)
