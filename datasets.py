import numpy as np
import os
import scipy.io
import torch
import synthetic_data_generator


def hsps_5():
    # TODO: Change to testing on train + test together?
    import h5py

    path = 'usps.h5'
    with h5py.File(path, 'r') as hf:
        train = hf.get('train')
        X = torch.from_numpy(train.get('data')[:]).float()
        y = torch.from_numpy(train.get('target')[:]).float()
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]
        return X, y, 'hsps_5'


def madelon():
    import pandas as pd
    # TODO: Add test set
    path_X = './/datasets//madelon_train.data'
    path_Y = './/datasets//madelon_train.labels'
    df_x = pd.read_csv(path_X, header=None, sep=' ')
    df_y = pd.read_csv(path_Y, header=None)
    X = df_x.iloc[:, :-1].values
    y = df_y.iloc[:].values
    return torch.from_numpy(X).float(), torch.from_numpy(y).flatten(), 'madelon',


datasets_funcs = [madelon]  # , hsps_5, synthetic_data_generator.get_synthetic_dataset]


def read_datasets(normalize=True):
    synth= [synthetic_data_generator.get_synthetic_dataset()]
    return synth
    # files = list(filter(lambda x: x.endswith('.mat'), os.listdir('datasets')))[1:]
    # for file in files:
    #     data = scipy.io.loadmat('.//datasets//' + file)
    #
    #     X, y = torch.from_numpy(data['X'].astype(np.float32)).float(), torch.from_numpy(data['Y']).flatten()
    #     name = file.split('.')[0]
    #     if normalize:
    #         X = (X - X.mean(dim=0)) / X.std(dim=0)
    #     yield X, y, name,{'n_relevant_features': 10}
