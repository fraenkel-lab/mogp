#!/usr/bin/env python3

import math
import numbers
import logging
from pathlib import Path
import joblib
import numpy as np
import numpy.random as npr
from sklearn.cluster import KMeans
from scipy.special import logsumexp as lse
from mogp.obsmodel import GPobs
from mogp.allocmodel import DPmix
from GPy.util.normalizer import Standardize

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - MOGP: %(levelname)s - %(message)s', "%I:%M:%S"))
logger.addHandler(handler)


class MoGP_constrained:
    def __init__(self, X, Y, alpha, num_init_clusters=5, num_iter=100, rand_seed=0, savepath='../../data/models/mogp_test',
                 savename='MoGP_constrained', save_iter=25, kernel='rbf', signal_variance=1.,
                 signal_variance_fix=True, noise_variance=1., noise_variance_fix=False, mean_func=True, threshold=None,
                 onset_anchor=True, normalize=False, Y_mean=None, Y_std=None):
        """A collapsed sampler for a DP mixture of GPs but with additional bias preferring monotonicity.

        Arguments:
            X (np.ndarray): explanatory variable (time since symptom onset) N * T (N - # patients, T - # time points)
            Y (np.ndarray): responses (clinical scores) N * T (N - # patients, T - # time points)
            alpha (float): Dirichlet Process scaling parameter; can influence the degree of cluster discretization and therefore the number of identified clusters
            num_init_clusters (int): number of initial clusters to assign with k-means initialization
            num_iter (int): number of iterations to run sampler
            rand_seed (int): random seed for k-means initialization
            savepath (str or pathlib.Path): directory in which to save model results
            savename (str): filename for model output
            save_iter (int): save model checkpoint recurrently after this number of iterations
            kernel (str): string specifying gaussian process kernel - implemented kernels limited to 'rbf' or 'linear'
            signal_variance (float): numeric value to initialize signal variance
            signal_variance_fix (bool): boolean flag for fixing signal (kernel) variance
            noise_variance (float): numeric value to initialize noise variance
            noise_variance_fix (bool): boolean flag for fixing noise variance
            mean_func (bool): boolean flag to indicate if negative linear mean function should be used (otherwise, mean function of 0 used)
            threshold (float or NoneType): numeric value to indicate strictness of monotonicity threshold
            onset_anchor (bool): boolean flag for if artificial onset anchor has been included in input data
        """
        # X and Y are assumed to be the same size and are organized as patients by time points
        self.X = X  # time course --- CAP scores in our case T * N
        self.Y = Y  # responses T * N (T - # time points, N - # patients)
        self.allocmodel = DPmix(alpha)  # DP mixture
        self.num_iter = num_iter
        self.num_init_clusters = num_init_clusters
        self.rand_seed = rand_seed
        self.savepath = Path(savepath)
        self.savename = savename
        self.save_iter = save_iter
        self.kernel = kernel
        self.mean_func = mean_func
        self.threshold = threshold

        self.signal_variance = signal_variance
        self.signal_variance_fix = signal_variance_fix
        self.noise_variance = noise_variance
        self.noise_variance_fix = noise_variance_fix
        self.onset_anchor = onset_anchor

        self.normalize = normalize
        self.mean = Y_mean
        self.std = Y_std

        self._check_validity_of_instance()
        self._initialize_normalizer()

        self.allocmodel.N, _ = X.shape
        self.z = np.zeros(self.allocmodel.N)  # stores which cluster the sample is in
        self.p = dict()  # stores the cluster probability
        self.lalloc = np.zeros([self.num_iter])
        self.lobs = np.zeros([self.num_iter])
        self.obsmodel = dict()
        self.occupancy = {}

        self.best_ll = -np.Inf
        self.ll = []
        self.burnin = 25

    def _check_validity_of_instance(self):
        """ Assert that the data passed to this program are valid, otherwise raise helpful error messages. """
        if not(isinstance(self.X, np.ndarray) & isinstance(self.Y, np.ndarray)):
            raise ValueError("X and Y must be np.ndarray. Wrong type passed in.")
        if not (self.X.shape == self.Y.shape):
            raise ValueError("X and Y must be the same size: {}, {}".format(self.X.shape, self.Y.shape))
        if not (isinstance(self.allocmodel.alpha, numbers.Number) and (self.allocmodel.alpha >= 0)):
            raise ValueError("parameter alpha must be a positive number. Was "+str(self.allocmodel.alpha))
        if not (isinstance(self.num_init_clusters, int) and (self.num_init_clusters > 0)):
            raise ValueError("number of initial clusters must be a positive integer greater than 0. Was "+str(self.num_init_clusters))
        if not (isinstance(self.num_iter, int) and (self.num_iter > 0)):
            raise ValueError("number of iterations must be a positive integer greater than 0. Was "+str(self.num_iter))
        if not (self.kernel in ['rbf', 'linear']):
            raise ValueError("Implemented kernels limited to 'rbf' or 'linear'. Was "+str(self.kernel))
        if not (isinstance(self.signal_variance, numbers.Number)):
            raise ValueError("parameter signal_variance must be number. Was " + str(self.signal_variance))
        if not (isinstance(self.signal_variance_fix, bool)):
            raise ValueError("parameter signal_variance_fix must be boolean. Was " + str(self.signal_variance_fix))
        if not (isinstance(self.noise_variance, numbers.Number)):
            raise ValueError("parameter noise_variance must be number. Was " + str(self.noise_variance))
        if not (isinstance(self.noise_variance_fix, bool)):
            raise ValueError("parameter noise_variance_fix must be boolean. Was " + str(self.noise_variance_fix))
        if not (isinstance(self.mean_func, bool)):
            raise ValueError("parameter mean_func must be boolean. Was " + str(self.mean_func))
        if not (isinstance(self.threshold, numbers.Number) or self.threshold is None):
            raise ValueError("parameter threshold must be None or number. Was " + str(self.threshold))
        if not (isinstance(self.onset_anchor, bool)):
            raise ValueError("parameter onset_anchor must be boolean. Was " + str(self.onset_anchor))
        if not (isinstance(self.normalize, bool)):
            raise ValueError("parameter normalize must be boolean. Was " + str(self.normalize))

        # Check normalizing inputs
        y_mean = np.mean(self.Y[~np.isnan(self.Y)])
        y_std = np.std(self.Y[~np.isnan(self.Y)])
        if y_std == 0:
            raise ValueError('standard deviation of values is 0; check inputs')
        if self.normalize is True:
            if (self.mean is not None) & (self.std is None):
                raise ValueError("both mean and std must be passed to be used for norm. missing std.")
            elif (self.mean is None) & (self.std is not None):
                raise ValueError("both mean and std must be passed to be used for norm. missing mean.")
        elif self.normalize is False:
            if (self.mean is not None) & (self.std is not None):
                raise ValueError('if parameter normalize is false, mean and std cannot be provided')
            elif not (math.isclose(y_mean, 0, abs_tol=0.1) & math.isclose(y_std, 1, rel_tol=0.1)):
                logger.warning("MoGP priors assume Y has been z-score normalized. mean was {:.2f} and std was {:.2f}. "
                               "recommend using normalize=True".format(y_mean, y_std))

    def _check_validity_of_predict_inputs(self, x_new, y_new):
        if not (isinstance(x_new, np.ndarray) & isinstance(self.Y, np.ndarray)):
            raise ValueError("x_new and y_new must be np.ndarray. Wrong type passed in.")
        if not (x_new.shape == y_new.shape):
            raise ValueError("x_new and y_new must be the same size: {} {}".format(x_new.shape, y_new.shape))

    def _initialize_normalizer(self):
        """Z-score input data by matrix mean; store original mean and std"""
        if self.normalize is True:
            if (self.mean is None) | (self.std is None):
                self.mean = np.mean(self.Y[~np.isnan(self.Y)])
                self.std = np.std(self.Y[~np.isnan(self.Y)])
            # update normalization
            self.Y = (self.Y-self.mean)/self.std
            # create normalizer obj for GPy model
            self.normalizer = Standardize()
            self.normalizer.mean = self.mean
            self.normalizer.std = self.std
            self.normalizer.std = self.std
        else:
            self.normalizer = None

    def _reset_normalizer(self):
        """Reset obsmodel to assume data is already normalized """
        active_comps_ids = np.where(self.allocmodel.Nk > 0)[0]
        for k in active_comps_ids:
            self.obsmodel[k].model.normalizer = None

    def _apply_normalizer(self):
        """Change normalizer so GP model prediction can use original data scale"""
        active_comps_ids = np.where(self.allocmodel.Nk > 0)[0]
        for k in active_comps_ids:
            self.obsmodel[k].model.normalizer = self.normalizer

    def _save_normalized_model(self, file_suffix):
        """Save model, with obsmodel that does NOT assume new input data has already been normalized"""
        self._apply_normalizer()
        joblib.dump(self, self.savepath / "{}_{}.pkl".format(self.savename, file_suffix))
        logger.info('Saved Model: {}'.format((self.savepath / "{}_{}.pkl".format(self.savename, file_suffix)).resolve()))
        self._reset_normalizer()

    def initialize_sampler(self, init_K, onset_anchor):
        """
        Initialize sampler with clusters using k-means clustering

        Arguments:
            init_K (int): number of initial clusters to use in k-means clustering
            onset_anchor (bool): flag to indicate if artificial anchor onset value has been included
        """

        np.random.seed(self.rand_seed)
        # If using data with onset anchor appended, kmeans clustering needs to be performed with second sample
        if onset_anchor:
            x_data = self.X[:, 1].reshape(-1, 1)
        else:
            x_data = self.X[:, 0].reshape(-1, 1)

        kmeans = KMeans(n_clusters=init_K, random_state=self.rand_seed).fit(x_data)
        self.z = kmeans.labels_
        for k in np.arange(init_K):
            Xk = self.X[self.z == k]
            Yk = self.Y[self.z == k]
            self.obsmodel[k] = GPobs(Xk[~np.isnan(Xk)].reshape(-1, 1), Yk[~np.isnan(Yk)].reshape(-1, 1),
                                     kernel=self.kernel, mean_func=self.mean_func,
                                     signal_variance=self.signal_variance, signal_variance_fix=self.signal_variance_fix,
                                     noise_variance=self.noise_variance, noise_variance_fix=self.noise_variance_fix)

    def sample(self):
        """Sampler for Mixture of Gaussian Process model"""

        self.initialize_sampler(init_K=self.num_init_clusters, onset_anchor=self.onset_anchor)
        self.allocmodel.calc_suffstats(self.z)
        idx = np.arange(self.allocmodel.N)
        logger.info('Cluster Initialization: {}'.format(self.allocmodel.Nk[self.allocmodel.Nk > 0]))

        # Dump initial model for debugging
        Path.mkdir(self.savepath, parents=True, exist_ok=True)
        self._save_normalized_model(file_suffix="iter{}".format("init"))

        for l in np.arange(1, self.num_iter):
            optimize_gp_params = True
            # Loop over all data
            for n in np.random.permutation(idx):
                curr_k = self.z[n]
                self.z[n] = -1  # Temporarily reset cluster for current individual
                # Remove data from current cluster and update its ll
                if self.X[self.z == curr_k].shape[0] == 0:
                    # Delete empty cluster
                    self.obsmodel[curr_k] = None
                else:
                    Xk = self.X[self.z == curr_k]
                    Yk = self.Y[self.z == curr_k]
                    self.obsmodel[curr_k].update_data(Xk[~np.isnan(Xk)].reshape(-1, 1),
                                                      Yk[~np.isnan(Yk)].reshape(-1, 1))
                # Compute conditional probability
                score, active_comps_ids = self.allocmodel.calc_score(curr_k)
                # Make new component
                new_comp_model = GPobs(self.X[n, ~np.isnan(self.X[n])].reshape(-1, 1),
                                       self.Y[n, ~np.isnan(self.Y[n])].reshape(-1, 1),
                                       kernel=self.kernel, mean_func=self.mean_func, signal_variance=self.signal_variance,
                                       signal_variance_fix=self.signal_variance_fix,
                                       noise_variance=self.noise_variance, noise_variance_fix = self.noise_variance_fix)

                for k in active_comps_ids[:-1]:
                    # Eliminate assignment to cluster if initial value for individual is above cluster initial value by threshold
                    if self.threshold is not None:
                        # If using data with onset anchor appended, set threshold based on second sample
                        index_to_eval = 1 if self.onset_anchor else 0

                        y_star, y_star_var = self.obsmodel[k].get_pred_at_loc(
                            self.X[n, ~np.isnan(self.X[n])].reshape(-1, 1), index_to_eval)
                        y_star_upper_bound = y_star  # y_star is the model prediction at the first X time point.

                        if np.any((self.Y[n, index_to_eval] - y_star_upper_bound) > self.threshold):
                            score[k] = -np.Inf  # Rule out cluster assignment if initial value is above threshold
                        else:
                            score[k] += self.obsmodel[k].calc_score(self.X[n, ~np.isnan(self.X[n])].reshape(-1, 1),
                                                                    self.Y[n, ~np.isnan(self.Y[n])].reshape(-1, 1))
                    else:
                        score[k] += self.obsmodel[k].calc_score(self.X[n, ~np.isnan(self.X[n])].reshape(-1, 1),
                                                                self.Y[n, ~np.isnan(self.Y[n])].reshape(-1, 1))
                new_comp = active_comps_ids[-1]
                score[new_comp] += new_comp_model.ll
                score = score[active_comps_ids]
                normalizing_constant = lse(score)

                # Resample according to CRP prior
                self.p[n] = np.exp(score - normalizing_constant)  # Should be numerically safe
                self.z[n] = npr.choice(active_comps_ids, p=self.p[n])
                self.allocmodel.update_suffstats(self.z[n])
                if self.z[n] == new_comp:
                    # New component created
                    self.obsmodel[self.z[n]] = new_comp_model
                else:
                    # Update stats of existing component
                    self.obsmodel[self.z[n]].update_suffstats(optimize_params=optimize_gp_params)

            logger.info('Iter {}: {}'.format(l, self.allocmodel.Nk[self.allocmodel.Nk > 0]))
            self.lalloc[l] = self.allocmodel.calc_ll()

            if l % self.save_iter == 0:
                # Save results every X iterations
                self._save_normalized_model(file_suffix="iter{}".format(l))

            active_comp_ids = np.where(self.allocmodel.Nk > 0)[0]
            for comp in active_comp_ids:
                self.lobs[l] += self.obsmodel[comp].calc_ll()

            if l > self.burnin:
                self.occupancy[l] = self.allocmodel.Nk[self.allocmodel.Nk > 0]
                # After burn in
                if (self.lobs[l] + self.lalloc[l]) > self.best_ll:
                    logger.info("At iteration {}, Previous Best {}, Current Best{}".format(l, self.best_ll, self.lobs[l] + self.lalloc[l]))
                    self.best_ll = self.lobs[l] + self.lalloc[l]

                    self._save_normalized_model(file_suffix="MAP")

        self.ll = self.lobs + self.lalloc

        self._apply_normalizer()
        joblib.dump(self, self.savepath / "{}.pkl".format(self.savename))
        logger.info("Saved Model: {}".format((self.savepath / "{}.pkl".format(self.savename)).resolve()))

    def predict(self, x_new, y_new):
        """ Predict cluster membership probabilities for new x and y vectors

        Arguments:
            x_new (np.ndarray): explanatory variable (time since symptom onset) for new patient (length: # time points)
            y_new (np.ndarray): responses (clinical scores) for new patient (length: # time points)
        Returns:
            p (np.ndarray): log likelihood scores for all cluster components, with the last value indicating ll score for a new individual component fit to the new data

        """
        self._check_validity_of_predict_inputs(x_new, y_new)

        # Account for normalization - normalize y_new, and reset model normalizer
        if self.normalize is True:
            y_new = (y_new-self.mean)/self.std
            # Reset obsmodel normalizers
            self._reset_normalizer()

        new_comp_model = GPobs(x_new.reshape(-1, 1), y_new.reshape(-1, 1), kernel=self.kernel,
                               signal_variance=self.signal_variance, signal_variance_fix=self.signal_variance_fix,
                               noise_variance=self.noise_variance, noise_variance_fix=self.noise_variance_fix)
        new_comp_model.model.optimize()

        score = np.log(np.hstack([self.allocmodel.Nk, self.allocmodel.alpha]) + 1e-16)
        active_comps_ids = np.where(self.allocmodel.Nk > 0)[0]
        for k in active_comps_ids:
            score[k] += self.obsmodel[k].calc_score(x_new.reshape(-1, 1), y_new.reshape(-1, 1))
        score[-1] += new_comp_model.ll
        normalizing_constant = lse(score)
        p = np.exp(score - normalizing_constant)

        if self.normalize is True:
            # Re-scale obsmodel to original scale
            self._apply_normalizer()

        return p
