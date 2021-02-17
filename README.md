# carrots

Is a machine learning library written in Python purely as an educational exercise.  

**THERE IS NO GOOD REASON WHATSOEVER TO USE CARROTS IN ANY KIND OF PRODUCTION ENVIRONMENT**

## Features
* Extremely simple model-based approach
* Univariate linear regression (multivariate in progress)
* Linear sample data generation (with or without random noise)
* That's about it right now

## Example implementation
```
from carrots import models, utils

train_x, train_y = utils.generate_linear_test_data(thetas=[4,3], num_datapoints=200, num_features=1, randomness=1)

lr = models.LinearRegression.LinearRegression(iterations=9999, learning_rate=0.01)
lr.fit(train_x, train_y, verbose=False)

predictions = [lr.predict(item) for item in train_x]
```

The model holds its own records for use in graphing and reporting
```
# User specified
model_object.learning_rate
model_object.iterations
model_object.lam (regularization lambda)

# Results of training
model_object.final_thetas
model_object.cost_history
model_object.theta_history
```
## Coming Soon
Multivariate Linear Regression
Each model will have built-in reporting via model_object.show_report()