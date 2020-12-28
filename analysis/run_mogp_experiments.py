import argparse
import multiprocessing
from pathlib import Path

import joblib
import numpy as np

from mogp import MoGP_constrained


class Experiment:
    def __init__(self, project, model_data_path, minnum, num_iter, expname=None, seed=None, kernel=None, multiprocess=False, normalize=True, y_mean=None, y_std=None):
        self.project = project
        self.model_data_path = model_data_path
        self.minnum = minnum
        self.expname = expname

        self.num_iter = np.int(num_iter)
        self.kernel = kernel
        self.seed = seed
        self.normalize = normalize
        self.y_mean = y_mean
        self.y_std = y_std

        self.multiprocess = multiprocess

        # Static model parameters - consistent across all experiments
        self.mean_func = True
        self.threshold = 0.5
        self.signal_variance = 1.
        self.signal_variance_fix = True
        self.noise_variance = 0.5
        self.noise_variance_fix = False

    def train_model(self):
        savepath = self.model_data_path / 'results' / self.kernel
        savename = 'model_{}_{}_{}_seed_{}'.format(self.project, self.minnum, self.expname, self.seed)
        data_dict = joblib.load(self.model_data_path / 'data_{}_{}_{}.pkl'.format(self.project, self.minnum, self.expname))

        XA = data_dict['XA']
        YA = data_dict['YA']
        num_patients = len(data_dict['SI'])
        num_init_clusters = round(num_patients / 50)
        alpha = num_init_clusters / np.log10(num_patients)


        mixR = MoGP_constrained(X=XA, Y=YA, alpha=alpha,
                                num_init_clusters=num_init_clusters, num_iter=self.num_iter, rand_seed=self.seed,
                                savepath=savepath, savename=savename, kernel=self.kernel,
                                mean_func=self.mean_func, threshold=self.threshold, signal_variance=self.signal_variance,
                                signal_variance_fix=self.signal_variance_fix, noise_variance=self.noise_variance,
                                noise_variance_fix=self.noise_variance_fix, normalize=self.normalize, Y_mean=self.y_mean, Y_std=self.y_std)
        mixR.sample()

    def run_experiment(self):
        assert (self.seed is not None and self.kernel is not None and self.expname is not None)
        """Enable multiprocessing"""
        if self.multiprocess is True:
            _ = multiprocessing.Process(target=self.train_model())
        else:
            self.train_model()


def full_alsfrst(project, num_iter=100, num_seeds=5, run_by_seed=False, seed=None, kernel=None, multiprocess=False):
    """ Run MoGP for all datasets (PROACT, AALS, CEFT, EMORY, GTAC)
        For PROACT: enable running seeds/kernels separately
        For all other datasets: run both rbf/linear kernels for 5 seeds
    """
    model_data_path = Path('data/model_data/1_alsfrsr_all/')
    expname = 'alsfrst'
    minnum = 'min3'

    curexp = Experiment(project=project, model_data_path=model_data_path, minnum=minnum, expname=expname,
                        num_iter=num_iter, multiprocess=multiprocess)

    if run_by_seed:
        assert (seed is not None) and (kernel is not None)
        curexp.seed = seed
        curexp.kernel = kernel
        curexp.run_experiment()

    else:
        kernels = ['rbf', 'linear']
        for cur_seed in range(0, num_seeds):
            for cur_kernel in kernels:
                curexp.seed = cur_seed
                curexp.kernel = cur_kernel
                curexp.run_experiment()


def alsfrst_predict(project, kernel, num_iter=100, num_seeds=5, multiprocess=False):
    """Run MoGP for prediction tasks - with option of using multiprocessing to speed compute"""

    model_data_path = Path('data/model_data/2_sparsity_prediction/prediction')
    minnum = 'min4'
    task_list = [0.25, 0.50, 1.0, 1.5, 2.0]

    curexp = Experiment(project=project, model_data_path=model_data_path, minnum=minnum, num_iter=num_iter, kernel=kernel, multiprocess=multiprocess)

    for cur_seed in range(0, num_seeds):
        for task in task_list:
            curexp.expname = 'predict_{}'.format(task)
            curexp.seed = cur_seed
            curexp.run_experiment()


def alsfrst_sparsity(project, kernel, num_iter=100, num_seeds=5,  multiprocess=False):
    """Run MoGP for prediction tasks - with option of using multiprocessing to speed compute"""

    model_data_path = Path('data/model_data/2_sparsity_prediction/sparsity')
    minnum = 'min10'
    task_list = [25, 50, 75]

    curexp = Experiment(project=project, model_data_path=model_data_path, minnum=minnum, num_iter=num_iter, kernel=kernel, multiprocess=multiprocess)

    for cur_seed in range(0, num_seeds):
        for task in task_list:
            curexp.expname = 'sparse_{}'.format(task)
            curexp.seed = cur_seed
            curexp.run_experiment()

def reference(project, num_iter=100, num_seeds=5, num_splits=5, run_by_seed=False, seed=None, multiprocess=False):
    """Run MoGP for reference model benchmarking"""
    model_data_path = Path('data/model_data/3_reference_transfer')
    kernel = 'rbf'
    minnum = 'min3'

    curexp = Experiment(project=project, model_data_path=model_data_path, minnum=minnum, num_iter=num_iter, kernel=kernel, multiprocess=multiprocess)

    if run_by_seed:
        assert (seed is not None)
        for split in range(0, num_splits):
            curexp.expname = 'alsfrst_train_split_{}'.format(split)
            curexp.seed = seed
            curexp.run_experiment()

    else:
        for cur_seed in range(0, num_seeds):
            for split in range(0,num_splits):
                curexp.expname = 'alsfrst_train_split_{}'.format(split)
                curexp.seed = cur_seed
                curexp.run_experiment()

def get_subscore_norm(model_data_path, project, minnum):
    """Normalize all alsfrs-r subscores with same norm factor"""
    Y_all = np.empty([1,1])
    for cur_task in ['alsfrst_bulb', 'alsfrst_fine', 'alsfrst_gross', 'alsfrst_resp']:
        cur_data = joblib.load(model_data_path / 'data_{}_{}_{}.pkl'.format(project, minnum, cur_task))
        print(np.mean(cur_data['YA'][~np.isnan(cur_data['YA'])]))
        Y_all = np.append(Y_all, cur_data['YA'])
    Y_mean = np.mean(Y_all[~np.isnan(Y_all)])
    Y_std = np.std(Y_all[~np.isnan(Y_all)])

    return Y_mean, Y_std

def alternate_outcomes(task, seed, num_iter=100, norm_consistent=False):
    """Run MoGP for alternate outcomes: ALSFRS-R Subscores and Forced Vital Capacity
        Tasks: alsfrst_bulb, alsfrst_fine, alsfrst_gross, alsfrst_resp, fvcpmax
    """
    model_data_path = Path('data/model_data/4_proact_alt_endpoints')
    project = 'proact'
    minnum = 'min3'
    kernel = 'rbf'
    expname = task

    curexp = Experiment(project=project, model_data_path=model_data_path, minnum=minnum, expname=expname, num_iter=num_iter, seed=seed, kernel=kernel)

    if norm_consistent:
        # Normalize all alsfrs-r scores to same ymean/ystd
        assert task is not 'fvcpmax'
        subsc_mean, subsc_std = get_subscore_norm(model_data_path, project, minnum)
        print(subsc_mean, subsc_std)
        curexp.Y_mean = subsc_mean
        curexp.Y_std = subsc_std

    curexp.run_experiment()


parser = argparse.ArgumentParser()
parser.add_argument("--exp", required=True, choices=['full', 'predict', 'sparse', 'ref', 'alt'])
parser.add_argument("--proj", default=None, choices=['aals', 'gtac', 'ceft', 'emory', 'proact'])
parser.add_argument("--kernel", default=None, choices=['rbf', 'linear'])
parser.add_argument("--num_iter", type=int, default=100)
parser.add_argument("--num_seeds", type=int, default=5)
parser.add_argument("--run_by_seed", default=False)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--multi", type=bool, default=False, choices=[True, False])
parser.add_argument("--numsplit", type=int, default=None)
parser.add_argument("--task", default=None, choices=['alsfrst_bulb', 'alsfrst_fine', 'alsfrst_gross', 'alsfrst_resp', 'fvcpmax'])
parser.add_argument("--norm_consistent", type=bool, default=None, choices=[True, False])


if __name__ == '__main__':
    args = parser.parse_args()
    if args.exp == 'full':
        full_alsfrst(project=args.proj, num_iter=args.num_iter, num_seeds=args.num_seeds,
                     run_by_seed=args.run_by_seed, seed=args.seed, kernel=args.kernel, multiprocess=args.multi)
    elif args.exp == 'predict':
        alsfrst_predict(project=args.proj, kernel=args.kernel, num_iter=args.num_iter, num_seeds=args.num_seeds, multiprocess=args.multi)

    elif args.exp == 'sparse':
        alsfrst_sparsity(project=args.project, kernel=args.kernel, num_iter=args.num_iter, num_seeds=args.num_seeds, multiprocess=args.multi)

    elif args.exp == 'ref':
        reference(project=args.prooject, num_iter=args.num_iter, num_seeds=args.num_seeds, num_splits=args.num_splits,
                  run_by_seed=args.run_by_seed, seed=args.esed, multiprocess=args.mult)
    elif args.exp == 'alt':
        alternate_outcomes(task=args.task, seed=args.seed, num_iter=args.num_iter, norm_consistent=args.norm_consistent)

    alternate_outcomes
