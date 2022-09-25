import numpy as np
import os
import scipy.io
import torch
import synthetic_data_generator


def mat_database(**kwargs):
    normalize = kwargs['normalize'] if 'normalize' in kwargs else True
    files = list(filter(lambda x: x.endswith('.mat'), os.listdir('datasets')))
    indexes = [2,4,6,7,8]
    filtered_files = [files[i] for i in indexes]
    for file in filtered_files:
        data = scipy.io.loadmat('.//datasets//' + file)

        X, y = torch.from_numpy(data['X'].astype(np.float32)).float(), torch.from_numpy(data['Y']).flatten()
        name = file.split('.')[0]
        if normalize:
            X = (X - X.mean(dim=0)) / X.std(dim=0)
        yield X, y, name, {'feature_amount': 20}


def one_mat_database_different_feature_amount(**kwargs):
    normalize = kwargs['normalize'] if 'normalize' in kwargs else True
    files = list(filter(lambda x: x.endswith('.mat'), os.listdir('datasets')))[1:]
    for file in files:
        data = scipy.io.loadmat('.//datasets//' + file)
        X, y = torch.from_numpy(data['X'].astype(np.float32)).float(), torch.from_numpy(data['Y']).flatten()
        name = file.split('.')[0]
        if normalize:
            X = (X - X.mean(dim=0)) / X.std(dim=0)
        yield X, y, name, {}

datset_groups = {
    'synthetic': synthetic_data_generator.get_synthetic_dataset,
    'real': mat_database,
    'real_one_database': one_mat_database_different_feature_amount
}


def read_datasets(group, **kwargs):
    return datset_groups[group](**kwargs)
