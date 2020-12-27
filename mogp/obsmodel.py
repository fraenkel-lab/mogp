import GPy as GPy
import numpy as np
from mogp.neg_linear import NegLinear


class GPobs:
    def __init__(self, X, Y, kernel='rbf', signal_variance=1., signal_variance_fix=True, noise_variance=1.,
                 noise_variance_fix=False, mean_func=True, lengthscale=4.):
        """
        Individual Gaussian Process (GP) model that contributes to the overall cluster model

        Arguments:
            X: independent variable of GP model, must be a vector
            Y: dependent variable of GP model, must be a vector
            kernel (str): type of kernel for the covariance model
            signal_variance (float): numeric value to initialize signal variance
            signal_variance_fix (bool): boolean flag for fixing signal (kernel) variance
            noise_variance (float): numeric value to initialize noise variance
            noise_variance_fix (bool): boolean flag for fixing noise variance
            mean_func (bool): boolean flag to indicate if negative linear mean function should be used (otherwise, mean function of 0 used)
            lengthscale (float): numeric value to control smoothness of rbf kernel
        """

        # Radial basis function (RBF) kernel; also known as squared-exponential kernel
        if kernel == 'rbf':
            # WARNING: assumes signal has been standardized (divided by standard deviation) and time is in years
            C = GPy.kern.RBF(input_dim=1, variance=signal_variance, lengthscale=lengthscale)
            C.lengthscale.set_prior(GPy.priors.Gamma.from_EV(4., 9.), warning=False)
            if signal_variance_fix:
                C.variance.fix()  # signal variance is fixed to constant
            else:
                C.variance.set_prior(GPy.priors.Gamma.from_EV(1., .5), warning=False)

        # Linear kernel with bias
        elif kernel == 'linear':
            # WARNING: assumes signal has been standardized (divided by standard deviation) and time is in years
            C = GPy.kern.Linear(input_dim=1, variances=signal_variance) + GPy.kern.Bias(1)
            if signal_variance_fix:
                C.linear.variances.fix()
            else:
                C.variances.set_prior(GPy.priors.Gamma.from_EV(1., .5), warning=False)

        # Set negative linear mean function
        if mean_func:
            mf = NegLinear(1, 1)
            self.model = GPy.models.GPRegression(X, Y, kernel=C, mean_function=mf, noise_var=noise_variance)
            self.model.mean_function.set_prior(GPy.priors.Gamma.from_EV(2 / 3, 0.2), warning=False)
        else:
            self.model = GPy.models.GPRegression(X, Y, kernel=C, noise_var=noise_variance)

        if noise_variance_fix:
            self.model.likelihood.variance.fix()
        else:
            self.model.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(0.75, 0.25**2), warning=False)

        self.ll = self.model.log_likelihood()
        self.X = X
        self.Y = Y
        self.tempX = None
        self.tempY = None
        self.templl = None

    def calc_ll(self):
        """ log probability of a partition """
        self.ll = self.model.log_likelihood()
        return self.ll

    def update_data(self, X, Y):
        """update_data is useful when removing data from the cluster. X and Y are the new data"""
        self.X = X
        self.Y = Y

        self.model.set_XY(X=self.X, Y=self.Y)

    def calc_score(self, Xnew, Ynew):
        """calculate log predictive density, with new sample"""
        self.tempX = np.vstack([self.X, Xnew])
        self.tempY = np.vstack([self.Y, Ynew])
        temp_metadata = None
        score = np.sum(self.model.log_predictive_density(Xnew, Ynew, temp_metadata))
        return score

    def get_pred_at_loc(self, Xnew, index=0):
        """predict ynew at xnew[index]; used for sampler initializing"""
        return self.model.predict(Xnew[index].reshape(-1, 1))

    def update_suffstats(self, optimize_params=True):
        """update sufficient statistics"""
        self.X = self.tempX
        self.Y = self.tempY

        self.model.set_XY(X=self.X, Y=self.Y)
        if optimize_params:
            self.model.optimize()
