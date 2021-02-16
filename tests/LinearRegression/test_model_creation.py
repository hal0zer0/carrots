import carrots

def test_model_creation():
    lr = carrots.models.LinearRegression.LinearRegression()

def test_model_attributes():
    lr = carrots.models.LinearRegression.LinearRegression()
    assert hasattr(lr, 'cost_history')
    assert hasattr(lr, 'final_thetas')
    assert hasattr(lr, 'theta_history')

def test_default_values():
    lr = carrots.models.LinearRegression.LinearRegression()
    assert lr.iterations == 500
    assert lr.learning_rate == 0.01
    assert lr.lam == 1
    assert lr.verbose is False

def test_passed_values():
    lr = carrots.models.LinearRegression.LinearRegression(iterations=42, learning_rate=0.05, lam=1.1)
    assert lr.iterations == 42
    assert lr.learning_rate == 0.05
    assert lr.lam == 1.1



