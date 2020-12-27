import numpy as np 
import scipy as sp


class DPmix:
    def __init__(self, alpha):
        """Dirichlet process (DP) mixture model"""
        self.alpha = alpha  # DP concentration prior
        self.Nk = None  # DP allocation sufficient statistics - number of participants in each active cluster
        self.N = None  # Number of patients

    def calc_ll(self):
        """Log probability of a partition under DP (specified by the Chinese Restaurant Prior (CRP))"""
        active_comps = self.Nk[self.Nk > 0]
        num_comp = active_comps.shape[0]
        return sp.special.gammaln(self.alpha) - sp.special.gammaln(self.N + self.alpha) + num_comp * np.log(self.alpha) + np.sum(sp.special.gammaln(active_comps))

    def calc_suffstats(self, z):
        """Calculate number of participants in each active cluster"""
        self.Nk = np.bincount(z)

    def calc_score(self, curr_k):
        """Returns both scores and active component ids"""
        self.Nk[curr_k] -= 1
        score = np.hstack([self.Nk, self.alpha])
        active_comp_ids = np.where(score > 0)[0]
        return np.log(score + 1e-16), active_comp_ids

    def update_suffstats(self, new_k):
        """Update number of participants in existing active cluster or add new cluster"""
        if new_k > self.Nk.shape[0]-1:
            self.Nk = np.hstack([self.Nk, 1])
        else:
            self.Nk[new_k] += 1
