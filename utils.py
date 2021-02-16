import numpy as np


def generate_linear_test_data(thetas, num_datapoints=100, num_features=1):
    train_x = 2 * np.random.rand(num_datapoints, num_features)
    # The /2 is just reducing the "randomness"
    train_y = thetas[0] + (thetas[1] * train_x + (np.random.randn(num_datapoints, 1) / 2))
    return train_x, train_y
