import numpy as np
from abc import abstractmethod
from model import Model
from metrics import mean_square_error, mean_absolute_error, r_squared


class LinearModel(Model):
    def __init__(self, fit_intercept=True, normalize=False, name=None):
        if name is not None:
            super(LinearModel, self).__init__(name)
        else:
            self.name = "LinearModelObject"
        self.fit_intercept = fit_intercept
        self.normalize = normalize

    @abstractmethod
    def fit(self):
        """ fit model """


class LinearRegression(LinearModel):
    def __init__(self, fit_intercept=True, normalize=False, name=None):
        if name is None:
            name = "LinearRegressionObject"
        super(LinearRegression, self).__init__(fit_intercept=fit_intercept, normalize=normalize, name=name)
        
        # coef_ should has shape [n_targets, n_features]
        self.coef_ = None
        # rank of coef_
        self.rank_ = None
        self.intercept = None

    def fit(self, X, y, sample_weight=None):
        """
        Fit linear model.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample
            .. versionadded:: 0.17
               parameter *sample_weight* support to LinearRegression.
        Returns
        -------
        self : returns an instance of self.
        """

        # check size of input
        y = y.reshape((-1, 1))
        assert X.shape[0] == y.shape[0]

        # compute coef_
        pseudo_inverse = np.linalg.inv(np.dot(X.T, X))
        pseudo_inverse = np.dot(pseudo_inverse, X.T)
        self.coef_ = np.dot(pseudo_inverse, y).T

    
    def _check_accuracy(self, X, y):
        y_ = np.dot(self.X, self.coef_)
        mse = mean_square_error(y_, y)
        text = "%20s: %8f"
        metrics = ["mean_square_error", "mean_absolute_error"]
        print("------Training Results------")
        print(text % "")
        



if __name__ == "__main__":
    model = LinearModel(name='my model')
    print(model)