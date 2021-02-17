import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=500, lam=1):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lam = lam
        self.verbose = False

        self.final_thetas = []
        self.cost_history = []
        self.theta_history = []

    @staticmethod
    def _cost(thetas, x, y):
        m = len(y)
        predictions = x.dot(thetas)
        cost = (1 / 2 * m) * np.sum(np.square(predictions - y))
        return cost

    def _update_thetas(self, x, y, thetas):
        m = len(y)
        prediction = np.dot(x, thetas)
        return (thetas * (1 - (self.learning_rate * (self.lam / m))) - (1 / m) * self.learning_rate * (
                x.T.dot((prediction - y))))

    def _gradient_descent(self, x, y, thetas, ):

        self.cost_history = np.zeros(self.iterations)
        self.theta_history = np.zeros((self.iterations, 2))  # adjust for multivariate?

        for it in range(self.iterations):

            thetas = self._update_thetas(x, y, thetas)
            self.theta_history[it, :] = thetas.T
            self.cost_history[it] = self._cost(thetas, x, y)
            if self.verbose:
                print("Iteration: {}\tCost: {}".format(it + 1, self.cost_history[it]))
                theta_string = " ".join(str(theta) for theta in thetas)
                print("Thetas for iteration {}: {}".format(it +1 , theta_string))

        return thetas, self.cost_history, self.theta_history

    def fit(self, x, y, autopad_x=False, verbose=True):
        """
        
        :param x:
        :param y:
        :param autopad_x:
        :param verbose:
        :return:
        """
        self.verbose = verbose

        thetas = np.random.randn(2, 1)  # adjust for multivariate
        if autopad_x:
            x = np.c_[np.ones((len(x), 1)), x]

        self.final_thetas, self.cost_history, self.theta_history = self._gradient_descent(x, y, thetas)

    def predict(self, x):
        return self.final_thetas[0] + x * self.final_thetas[1]
