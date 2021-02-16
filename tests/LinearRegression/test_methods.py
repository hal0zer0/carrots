import carrots
import numpy as np
import random


def test_cost_function():
    """
    We should be able to specify 'any' theta values here, and generate perfectly linear data on them
    So that when we then run the cost function against the data, we get exactly 0

    """
    t0 = random.randint(-5, 6)
    t1 = random.randint(-5, 6)

    thetas = [[t0], [t1]]
    x, y = carrots.utils.generate_linear_test_data(thetas, randomness=0, one_pad_x=True)


    lr = carrots.models.LinearRegression.LinearRegression()

    assert lr._cost(thetas, x, y) == 0
