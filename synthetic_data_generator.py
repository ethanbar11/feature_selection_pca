import matplotlib.pyplot as plt
import numpy as np
import torch


def create_data(n_classes, min_points_per_class,
                max_points_per_class, n_relevant_features, n_false_feature, initial_mu, SD):
    """Create a synthetic dataset with n_classes classes.
    Each class has a number of points between min_points_per_class
    and max_points_per_class.
    Each point has n_relevant_features features that are relevant
    for the class and n_false_feature features that are not relevant
    for the class.
    """
    # Create the classes
    n_features = n_relevant_features + n_false_feature
    X = None
    y = None
    for i in range(n_classes):
        # Create the points for the class
        n_points = np.random.randint(min_points_per_class,
                                     max_points_per_class)
        mu = np.random.rand(n_features) * (2 * initial_mu) - initial_mu  # random vector in [-2000,2000]
        sigma = np.ones(n_features) * (np.random.rand() + 1) * SD  # random vector in [0,2]
        to_add_x = np.random.normal(mu, sigma, (n_points, n_features))
        to_add_y = np.ones(n_points) * i
        if X is not None:
            X = np.append(X, to_add_x, axis=0)
            y = np.append(y, to_add_y, axis=0)

        else:
            X = to_add_x
            y = to_add_y

    # Random permutation of false features
    for feature_index in range(n_relevant_features, n_features):
        X[:, feature_index] = np.random.permutation(X[:, feature_index])
    return X, y


def show_data(X, y):
    # Show color data using imagesec equivalent
    plt.imshow(X, aspect='auto', interpolation='none')
    plt.show()


def save_data(X, y):
    np.save('.//data//X.npy', X)
    np.save('.//data//y.npy', y)


def get_synthetic_dataset(seed=42, times=1):
    for i in range(times):
        np.random.seed(seed + i)
        n_classes = np.random.randint(2, 8)
        min_points_per_class = 150
        max_points_per_class = 250
        n_relevant_features = np.random.randint(5, 100)
        n_false_feature = np.random.randint(30, 400)
        n_false_feature = 200
        mu = 1
        SD = np.random.randint(1, 10) * 0.01
        # print('Creating data')
        # print(
        #     'Initial params are : n_classes = {}, min_points_per_class = {}, max_points_per_class = {}, n_relevant_features = {}, n_false_feature = {}, mu = {}, SD = {}'
        #     .format(n_classes, min_points_per_class, max_points_per_class, n_relevant_features, n_false_feature, mu,
        #             SD))
        X, y = create_data(n_classes, min_points_per_class,
                           max_points_per_class, n_relevant_features, n_false_feature, mu, SD)
        metadata = {'n_classes': n_classes,
                    'dataset_size': X.shape[0],
                    'min_points_per_class': min_points_per_class,
                    'max_points_per_class': max_points_per_class,
                    'n_relevant_features': n_relevant_features,
                    'n_false_feature': n_false_feature,
                    'mu': mu,
                    'SD': SD}
        yield torch.from_numpy(X).float(), torch.from_numpy(y), 'Synthetic Dataset seed {}'.format(seed + i), metadata
