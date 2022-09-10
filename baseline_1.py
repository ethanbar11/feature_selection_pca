import torch
import torch.optim
import sklearn.cluster
from sklearn.metrics.cluster import normalized_mutual_info_score


class FeatureExtractionAlgorithm:
    # Expecting X to be numpy array of size (n_samples, n_features)
    def get_relevant_features(self, X):
        raise NotImplementedError


class Baseline(FeatureExtractionAlgorithm):
    def get_relevant_features(self, X):
        return [i for i in range(X.shape[1])]


# This is specific to the synthetic data generator
class Perfect(FeatureExtractionAlgorithm):
    def get_relevant_features(self, X):
        return [i for i in range(X.shape[1]) if i < 5]


def calculate_compression(u, v, P):
    diff = u - v
    outcome = torch.linalg.norm(diff) / torch.linalg.norm(torch.matmul(P, diff))
    return outcome


def calculate_compression_w(u, v, P, w):
    weighted_diff = w * (u - v)
    weighted_diff_P = torch.matmul(weighted_diff, P.T)
    outcome = torch.linalg.norm(weighted_diff, dim=1) / torch.linalg.norm(weighted_diff_P, dim=1)
    return outcome


def optimize_w(B, P, w=None):
    # B is a matrix of size (chi * n**2 , d)
    # w is a vector of size (d, 1)
    # P is a matrix of size (k, d)
    # Split B into batches of rows
    BATCH_SIZE = B.shape[0]
    LEARNING_RATE = 0.1
    EPOCHS = 100
    n = B.shape[0] // BATCH_SIZE
    B_batches = torch.split(B, BATCH_SIZE, dim=0)
    if w is None:
        # Adding requires grad here to make sure that we can calculate the gradient
        # of w
        # TODO: Maybe change initialization / add constraints
        w = torch.ones((B.shape[-1]), requires_grad=True)
    loss = None
    optimizer = torch.optim.SGD(params=[w], lr=LEARNING_RATE)#,weight_decay=0.01)
    w_history = []
    loss_history = []
    for epoch in range(EPOCHS):
        for batch in B_batches:
            # Calculating the gradient of the loss function
            # with respect to w
            optimizer.zero_grad()
            u = batch[:, 0, :]
            v = batch[:, 1, :]

            # TODO: Maybe change mean to sum or other form.
            loss = torch.mean(calculate_compression_w(u, v, P, w))
            loss *= -1
            loss.backward()
            optimizer.step()
        sorted_w, indices = torch.sort(w, descending=True)
        maximal_w_indices = indices[:20]
        maximal_good_w_indices = filter(lambda x: x < 20, maximal_w_indices)
        # print('Epoch : {} Loss : {}'.format(epoch, loss))
        # print('W ratio', len(list(maximal_good_w_indices)) / len(maximal_w_indices))
        w_history.append(len(list(maximal_good_w_indices)) / len(maximal_w_indices))
        val = loss.detach().numpy()
        loss_history.append(val.item(0))
        # print('W indexes:', indices[:20])
        # print('W values:', sorted_w[:20])

    # Plot w history
    import matplotlib.pyplot as plt
    loss_history = torch.tensor(loss_history) / torch.mean(torch.tensor(loss_history))
    w_history = torch.tensor(w_history) / torch.mean(torch.tensor(w_history))
    plt.plot(w_history)
    plt.plot(loss_history)
    plt.show()


class PCAFeatureExtraction(FeatureExtractionAlgorithm):
    def __init__(self, n_components):
        self.n_components = n_components
        self.xi = 0.01  # between [0,1], size of B out of n^2 pairs

    def get_relevant_features(self, X):
        pass
        # pca = sklearn.decomposition.PCA(n_components=self.n_components)
        # pca.fit(X)
        # return pca.components_

    def train(self, X, y=None):
        # First stage - create all pairs of vectors
        n = X.shape[0]
        indices = torch.tensor([i for i in range(n)])
        left_indices, right_indices = torch.meshgrid(indices, indices)

        left_indices = left_indices.flatten()
        right_indices = right_indices.flatten()
        left = torch.index_select(X, 0, left_indices)
        right = torch.index_select(X, 0, right_indices)

        left_y = torch.index_select(y, 0, left_indices)
        right_y = torch.index_select(y, 0, right_indices)

        pairs = torch.stack((left, right), dim=1)

        # Second stage - For each pair - calculate compressibility ratio using PCA

        # TODO: Check what Lior said with not needing to substract the mu
        pca = sklearn.decomposition.PCA(n_components=self.n_components)
        pca.fit(X)
        P = torch.from_numpy(pca.components_)  # Should be sized (n_components, n_features)

        grades = []
        # TODO: Perform this whole operation together with removing duplicates
        # pairs before, and then just calculate the grades.
        for i in range(n ** 2):
            if not np.array_equal(pairs[i][0], pairs[i][1]):
                grade = calculate_compression(pairs[i][0], pairs[i][1], P)
                same_class = left_y[i] == right_y[i]
                grades.append((grade, same_class, pairs[i]))
        # Still second - Create B - the group of top xi pairs according to measure
        grades = sorted(grades, key=lambda x: x[0], reverse=True)
        B = grades[:int(self.xi * n ** 2)]
        print('Same group precentage in B is {}'.format(sum([x[1] for x in B]) / len(B)))
        print('Same group precentage in all pairs is {}'.format(sum([x[1] for x in grades]) / len(grades)))
        B = torch.stack([x[2] for x in B], dim=0)
        # Third stage - create w and Perform SGD on w where the loss
        # is -1 * mean(compressibility of batch)

        # optimize_w(B, P)

        # Fourth stage - save w


# Running k-means and returning accuracy based on algo.
def run_algo(algo, X, y, seed=0):
    NUM_OF_RUNS = 5
    features = algo.get_relevant_features(X)
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
    import numpy as np

    X = torch.from_numpy(np.load('.//data//X.npy'))
    y = torch.from_numpy(np.load('.//data//y.npy'))
    # algo = Baseline()
    # run_algo(algo, X, y)
    # exit()
    for pca_k in range(3, 19):
        print("PCA_k", pca_k)
        algo = PCAFeatureExtraction(pca_k)
        algo.train(X, y)
