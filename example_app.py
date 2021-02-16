from carrots import models, utils
from matplotlib import pyplot as plt

# Keep num_features at 1, MULTIVARIATE NOT YET WORKING
train_x, train_y = utils.generate_linear_test_data(thetas=[4,3], num_datapoints=200, num_features=1, randomness=1)

lr = models.LinearRegression.LinearRegression(iterations=9999, learning_rate=0.01)
lr.fit(train_x, train_y, verbose=False)

predictions = [lr.predict(item) for item in train_x]

plt.scatter(train_x, train_y)
plt.plot(train_x, predictions, color='red')
plt.show()
