import scipy.io as sio
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def getData(filename):
    load_data = sio.loadmat(filename)
    y = load_data['y']
    X = load_data['X'].transpose(3, 0, 1, 2)
    return X, y


if __name__ == '__main__':
    file = 'data/train_32x32.mat'
    X, y = getData(filename=file)
    print(X.shape)