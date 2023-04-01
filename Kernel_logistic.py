import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from funclog import Newton

class LogisticRegressor(BaseEstimator, ClassifierMixin):

    def __init__(self, lamb=1.):
        """
        This class implements methods for fitting and predicting with a LogesticRegression for classification
        inputs:
        - lamb : the regularisation parameter
        """
        self.lamb = lamb

    def fit(self, X, y):
        """
        inputs:
        - X (size: Nxd): the points we want to classify
        - y (size: Nx1): the values of the classes
        outputs:
        - the value of MLE estimation (w_hat, b_hat) in the Linear regression model
        """
        w0, b0 = np.random.randn(1, 100)*0.07, np.zeros((1,1))
        self.w_, self.b_, _ = Newton(X, y, w0, b0)

        return self

    def predict(self, X):
        """
        inputs:
        - X (size Nxd): a point in R^d
        - w (size: 1xd): the weights of the affine mapping of x
        - b (size: 1x1): the constant of the affine mapping of x
        output:
         - the predicted class for the associated y given the
        Linear Regression parameters
        """
        return (self.w_@X.T + self.b_ > 1/2).astype("int")

    def score(self, X, y):
        """
        inputs:
        - X (size Nxd): the points in R^d we want to classify
        - y (size Nx1): the labels of the points
        """
        y_pred = self.predict(X)
        return np.sum(y_pred == y)/y.shape[0]
