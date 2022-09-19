class FeatureExtractionAlgorithm:
    def __init__(self, **kwargs):
        self.results_handler = kwargs['results_handler']

    # Expecting X to be numpy array of size (n_samples, n_features)
    def get_relevant_features(self, X):
        raise NotImplementedError

    # def run_k_means(self, X, y):
    #     for n_features in self.feature_amount:
    #         NUM_OF_RUNS = 5
    #         features = self.get_relevant_features(X, n_features)
    #         n_clusters = len(torch.unique(y))
    #         X_used = X[:, features]
    #         accuracies = []
    #         seed = 42
    #         for i in range(NUM_OF_RUNS):
    #             kmeans = KMeans(n_clusters=n_clusters, random_state=seed + i).fit(X_used.cpu())
    #             y_pred = kmeans.labels_
    #             mutual_info_score = normalized_mutual_info_score(y, y_pred)
    #             accuracies.append(mutual_info_score)
    #         self.results_handler.add_result('NMI', (n_features, torch.mean(torch.tensor(accuracies))))

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
