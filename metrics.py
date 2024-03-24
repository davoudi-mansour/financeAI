import numpy as np

def mse(y_true, y_pred, reduction=np.mean, axis=None):
    return reduction((np.array(y_true) - np.array(y_pred)) ** 2, axis=axis)


def rmse(y_true, y_pred, reduction=np.mean, axis=None):
    return np.sqrt(mse(y_true, y_pred, reduction, axis=axis))


def mae(y_true, y_pred, reduction=np.mean, axis=None):
    return reduction(np.abs(y_true - y_pred), axis=axis)


def mape(y_true, y_pred, reduction=np.mean, axis=None):
    return 100.0 * reduction(np.abs((y_true - y_pred) / y_true), axis=axis)


def smape(y_true, y_pred, reduction=np.mean, axis=None):
    return 200.0 * reduction(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)), axis=axis)
