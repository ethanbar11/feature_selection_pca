import matplotlib.pyplot as plt
import numpy as np


def create_data(n_classes, min_points_per_class,
                max_points_per_class, n_relevant_features, n_false_feature, SD):
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
        mu = np.random.rand(n_features) * 2 - 1  # random vector in [-1,1]
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
    for feature_index in range(n_relevant_features + 1, n_features):
        X[:, feature_index] = np.random.permutation(X[:, feature_index])

    return X, y


def show_data(X, y):
    # Show color data using imagesec equivalent
    plt.imshow(X, aspect='auto', interpolation='none')
    plt.show()


def save_data(X, y):
    np.save('.//data//X.npy', X)
    np.save('.//data//y.npy', y)


if __name__ == '__main__':
    n_classes = 10
    min_points_per_class = 5
    max_points_per_class = 40
    n_relevant_features = 20
    n_false_feature = 150
    SD = 1 / 100
    X, y = create_data(n_classes, min_points_per_class,
                       max_points_per_class, n_relevant_features, n_false_feature, SD)
    save_data(X, y)
    exit()
