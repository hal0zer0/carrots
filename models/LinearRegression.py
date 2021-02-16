import numpy as np


class LinearRegression:
    def __init__(self):
        self.final_thetas = []
        self.cost_history = []
        self.theta_history = np.array([])

    def _cost(self, thetas, x, y):
        m = len(y)
        predictions = x.dot(thetas)
        cost = (1/2 * m) * np.sum(np.square(predictions - y))
        return cost

    def _gradient_descent(self, x, y, thetas, learning_rate=0.01, iterations=500, lam=1, verbose=True):
        m = len(y)
        cost_history = np.zeros(iterations)
        theta_history = np.zeros((iterations, 2))

        for it in range(iterations):
            prediction = np.dot(x, thetas)
            thetas = (thetas * (1 - (learning_rate * (lam/m))) - (1/m)*learning_rate*(x.T.dot((prediction - y))))
            theta_history[it,:] = thetas.T
            cost_history[it] = self._cost(thetas, x, y)
            if verbose:
                print("Iteration: {}\tCost: {}".format(it, cost_history[it]))
                print(thetas)

        return thetas, cost_history, theta_history

    def fit(self, x, y):
        thetas = np.random.randn(2,1)
        X_b = np.c_[np.ones((len(x), 1)), x]
        self.final_thetas, self.cost_history, self.theta_history = self._gradient_descent(X_b, y, thetas)

    def predict(self, x):
        return self.final_thetas[0] + x * self.final_thetas[1]
