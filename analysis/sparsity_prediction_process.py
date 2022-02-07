#!/usr/bin/env python3

from analysis_utils import *

import joblib
import numpy as np
from pathlib import Path

import argparse


def calc_y_pred_spec_data(data_box, data_spec_box, i, model_type=None, mod=None):
    """Calculate y_prediction with sparse data matrices"""
    full_set = data_box['XA'][i]
    x_train = data_spec_box['XA'][i]
    x_train = x_train[~np.isnan(x_train)]
    y_train = data_spec_box['YA'][i]
    y_train = y_train[~np.isnan(y_train)]

    ind_diff = [j for j, item in enumerate(full_set) if ((item not in x_train) and (~np.isnan(item)))]

    x_test = data_box['XA'][i][ind_diff]  # Data withheld from model training
    y_test = data_box['YA'][i][ind_diff]  # Data withheld from model training

    y_pred = calc_y_pred(model_type=model_type, x_train=x_train, y_train=y_train,
                                 x_pred=x_test, mod=mod, i=i)
    assert y_test.shape == y_pred.shape, 'prediction and original y vectors not same shape'

    return y_test, y_pred


def calc_mogp_rmse(model_type, data_path, project, min_num, task_name, mat_opts, results_path=None, alphasc=None):
    """Calcluate rmse error between witheld data and mean function of model"""
    assert model_type in ['slope', 'sigmoid', 'rbf', 'linear'], 'model type {} not implemented'.format(
        model_type)  # all currently implemented baselines

    full_data = joblib.load(data_path / 'data_{}_{}_{}_full.pkl'.format(project, min_num, task_name))
    mod_obj = {}
    for mat in mat_opts:
        name = '{}_{}_{}_{}'.format(project, min_num, task_name, mat)
        print('{}: {}'.format(name, model_type))
        cur_data = joblib.load(data_path / 'data_{}.pkl'.format(name))

        num_patients = len(cur_data['SI'])
        err_arr = np.zeros((num_patients, 1))
        si = cur_data['SI']
        mod_type = model_type

        if model_type in ['rbf', 'linear']:
            assert results_path is not None

            if alphasc is None:
                cur_model = get_map_model(results_path, 'model_{}'.format(name))
            else:
                cur_model = get_map_model(results_path, 'model_{}_alphasc_{}'.format(name, alphasc)) #add alphasc flag

            assert len(cur_model.z) == len(cur_data['SI']), 'Number of participants in data and model do not match'
            assert model_type == cur_model.kernel, 'model type and model kernel do not match'
            num_clust = len(np.where(cur_model.allocmodel.Nk > 0)[0])
            seed = cur_model.rand_seed
            best_ll = cur_model.best_ll
            dict_name = '{}_seed_{}'.format(name, seed)

        elif model_type in ['slope', 'sigmoid']:
            cur_model = None
            num_clust = num_patients
            seed = np.nan
            best_ll = np.nan
            dict_name = name

        # elif model_type in ['lme']:
        #     cur_model = train_lme_model(cur_data)
        #     num_clust = num_patients
        #     seed = np.nan
        #     best_ll = np.nan
        #     dict_name = name

        for i in range(0, num_patients):
            y_real, y_pred_mean = calc_y_pred_spec_data(full_data, cur_data, i, model_type=model_type, mod=cur_model)
            err_arr[i] = calc_error(y_real, y_pred_mean)

        mod_obj[dict_name] = ModelSum(task_name, mat, seed, mod_type, err_arr, num_clust, best_ll, si)
    return mod_obj


def gen_mod_obj_full(project, task_name, save=False):
    """Generate model objects for sparsity and prediction experiments"""
    exp_path = Path('data/model_data/2_sparsity_prediction')

    mod_obj_dict = {}
    if task_name == 'predict':
        min_num = 'min4'
        mat_opts = ['0.25', '0.5', '1.0', '1.5', '2.0']
        data_path = exp_path / 'prediction'

    elif task_name == 'sparse':
        min_num = 'min10'
        mat_opts = ['25', '50', '75']
        data_path = exp_path / 'sparsity'

    for base_mod in ['rbf',  'linear']:
        res_path = data_path / 'results' / base_mod
        mod_obj_dict[base_mod] = calc_mogp_rmse(base_mod, data_path, project, min_num, task_name, mat_opts,
                                     results_path=res_path)

    for base_mod in ['slope', 'sigmoid']:
        mod_obj_dict[base_mod] = calc_mogp_rmse(base_mod, data_path, project, min_num, task_name, mat_opts)

    if save:
        save_path = data_path / 'results' / 'rmse'
        Path.mkdir(save_path, parents=True, exist_ok=True)
        print(save_path.resolve())
        for key in mod_obj_dict.keys():
            joblib.dump(mod_obj_dict[key], save_path / '{}_{}_{}_rmse_err.pkl'.format(task_name, project, key))

    return mod_obj_dict

def gen_mod_obj_full_alpha(project, task_name, save=False, alphasc=None):
    """Generate model objects for sparsity and prediction experiments"""
    exp_path = Path('data/model_data/2_sparsity_prediction')

    mod_obj_dict = {}
    if task_name == 'predict':
        min_num = 'min4'
        mat_opts = ['0.25', '0.5', '1.0', '1.5', '2.0']
        data_path = exp_path / 'prediction'

    # elif task_name == 'sparse':
    #     min_num = 'min10'
    #     mat_opts = ['25', '50', '75']
    #     data_path = exp_path / 'sparsity'

    for base_mod in ['rbf',  'linear']:
        res_path = data_path / 'results' / base_mod
        mod_obj_dict[base_mod] = calc_mogp_rmse(base_mod, data_path, project, min_num, task_name, mat_opts,
                                     results_path=res_path, alphasc=alphasc)

    # for base_mod in ['slope', 'sigmoid']:
    #     mod_obj_dict[base_mod] = calc_mogp_rmse(base_mod, data_path, project, min_num, task_name, mat_opts)

    if save:
        save_path = data_path / 'results' / 'rmse'
        Path.mkdir(save_path, parents=True, exist_ok=True)
        print(save_path.resolve())
        for key in mod_obj_dict.keys():
            joblib.dump(mod_obj_dict[key], save_path / '{}_{}_{}_{}_rmse_err.pkl'.format(task_name, project, key, alphasc))

    return mod_obj_dict

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=bool, default=False, choices=[True, False])

if __name__ == "__main__":
    args = parser.parse_args()
    if args.alpha:
        alphsc_list = [0.1, 0.5, 2, 10]
        for a in alphsc_list:
            print('calculating_alpha_rmse: ', a)
            _ = gen_mod_obj_full_alpha('ceft', 'predict', save=True, alphasc=a)
    else:
        # calculates RMSE error for prediction/sparsity experiments between witheld data and trajectory mean function
        _ = gen_mod_obj_full('ceft', 'sparse', save=True)
        _ = gen_mod_obj_full('ceft', 'predict', save=True)
        _ = gen_mod_obj_full('proact', 'sparse', save=True)
        _ = gen_mod_obj_full('proact', 'predict', save=True)
