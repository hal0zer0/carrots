import numpy as np


def generate_linear_test_data(thetas, num_datapoints=100, num_features=1, randomness=1, one_pad_x=True):
    """
    Generates x and y data suitable to be passed directly into carrots
    :param thetas: thetas[0] is the bias unit, others are threated as normal feature thetas
    :param num_datapoints: How many data points to generate
    :param num_features: NOT YET IMPLEMENTED past 1
    :param randomness: 0 for no random influence on the data
    :param one_pad_x: boolean - whether or not to prepend a column of 1s to the x data table
    :return:
    """
    train_x = np.random.rand(num_datapoints, num_features)

    variance = np.random.randn(num_datapoints, 1) /2 * randomness

    train_y = thetas[0] + (thetas[1] * train_x + variance)
    if one_pad_x:
        train_x = np.c_[np.ones((len(train_x), 1)), train_x]
    return train_x, train_y
