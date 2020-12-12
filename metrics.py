import numpy as np


def shape_squeeze(y_, y):
    try:
        assert y_.shape == y.shape
    except:
        raise Exception("Sizes don't fit.")

    if y.ndim > 1:
        y = y.reshape(-1)
        y_ = y_.reshape(-1)

    return y_, y


def mean_square_error(y_, y):
    y_, y = shape_squeeze(y_, y)

    n_samples = y.shape[0]
    
    metrics = np.sum((y_ - y)**2)
    metrics = metrics / n_samples
    
    return metrics


def mean_absolute_error(y_, y):
    y_, y = shape_squeeze(y_, y)

    n_samples = y.shape[0]
    
    metrics = np.sum(np.abs(y_ - y))
    metrics = metrics / n_samples
    
    return metrics


def r_squared(y_true, y_pred):
    y_true, y_pred = shape_squeeze(y_true, y_pred)
    variance = np.var(y_pred)
    mse = mean_square_error(y_pred, y_true)
    return 1 - mse / variance


if __name__ == "__main__":
    y = np.random.random((100, 1))
    y_ = np.random.random((100, 1))

    print(mean_square_error(y_, y))
    print(mean_absolute_error(y_, y))
    print(r_squared(y_, y))