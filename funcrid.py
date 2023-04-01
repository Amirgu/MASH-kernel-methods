
### ===== functions for Ridge Regression ===== ###

import numpy as np

def compute_RR_MLE(X,y,lamb):
    """
    inputs:
    - X (size: Nxd): the points we want to classify
    - y (size: Nx1): the values of the classes
    outputs:
    - the value of MLE estimation (w_hat, b_hat) in the Linear regression model
    """
    X_tilde = np.vstack((X,np.ones(X.shape[1])))
    temp = np.linalg.inv(X_tilde@X_tilde.T + lamb*X.shape[1]*np.eye(1+X.shape[0]))@X_tilde@y.T
    return temp[:-1], temp[-1]

def predict_RR(x,w,b):
    """
    inputs:
    - x (size 1xd): a point in R^d
    - w (size: 1xd): the weights of the affine mapping of x
    - b (size: 1x1): the constant of the affine mapping of x
    output:
     - the predicted class for the associated y given the
    Linear Regression parameters
    """
    return (w.T@x+b>1/2).astype("int")

### ===== functions for Kernel Ridge Regression ===== ###

def compute_KRR_MLE(X, y, lamb, sig=10):
    """
    inputs:
    - X (size: N_trxd): the points of the training set
    - y (size: N_trx1): the values of the classes
    outputs:
    - the value of MLE estimation (w_hat, b_hat) in the kernel ridge regression model
    """
    K = Gaussian_kernel(X, X, sig=sig)
    alpha = np.linalg.inv(K+lamb*X.shape[1]*np.eye(X.shape[1]))@y.T
    return alpha

def predict_KRR(X_tr, X_te, alpha, sig=10):
    """
    inputs:
    - X_tr (size N_trxd): the points of the training set
    - X_te (size N_texd): the points of the test set we want to classify
    - w (size: 1xd): the weights of the affine mapping
    - b (size: 1x1): the constant of the affine mapping
    output:
     - the predicted class for the associated y_te given the
    Linear Regression parameters
    """
    K_te_tr = Gaussian_kernel(X_tr, X_te, sig=sig)
    return 2*(alpha.T@K_te_tr>0).astype("int")-1