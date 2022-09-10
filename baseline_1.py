import torch
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


class PCAFeatureExtraction(FeatureExtractionAlgorithm):
    def __init__(self, n_components):
        self.n_components = n_components
        self.xi = 0.05  # between [0,1], size of B out of data

    def get_relevant_features(self, X):
        pass
        # pca = sklearn.decomposition.PCA(n_components=self.n_components)
        # pca.fit(X)
        # return pca.components_

    def calculate_B(self, X):
        pass

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
                grades.append((grade, same_class))
        # Still second - Create B - the group of top xi pairs according to measure
        grades_only_same_class = np.mean(np.array([x[0] for x in grades if x[1]]))
        grades_only_different_class = np.mean(np.array([x[0] for x in grades if not x[1]]))
        print("grades_only_same_class", grades_only_same_class)
        print("grades_only_different_class", grades_only_different_class)
        print('ratio', grades_only_same_class / grades_only_different_class)
        # Third stage - create w and Perform SGD on w where the loss
        # is -1 * mean(compressibility of batch)

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
    for pca_k in range(1, 19):
        print("PCA_k", pca_k)
        algo = PCAFeatureExtraction(pca_k)
        algo.train(X, y)
