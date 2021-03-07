#!/usr/bin/env python3

from analysis_utils import *

import joblib
import numpy as np
from pathlib import Path


def calc_y_pred_spec_data(data_box, data_spec_box, i, slope=False, model_box=None):
    """Calculate y_prediction with either mogp model or slope"""
    full_set = data_box['XA'][i]
    pred_set = data_spec_box['XA'][i]
    ind_diff = [j for j, item in enumerate(full_set) if ((item not in pred_set) and (~np.isnan(item)))]
    x_val = data_box['XA'][i][ind_diff]
    y_real = data_box['YA'][i][ind_diff]

    if slope:
        y_pred = pred_linreg_fxn(data_spec_box['XA'][i], data_spec_box['YA'][i], x_val)
    else:
        cur_clust = model_box.z[i]
        y_pred = calc_y_pred_model(model_box, x_val, cur_clust)

    return y_real, y_pred


def calc_mogp_rmse(data_path, results_path, project, min_num, task_name, mat_opts):
    """Calcluate rmse error between witheld data and mean function of model"""
    full_data = joblib.load(data_path / 'data_{}_{}_{}_full.pkl'.format(project, min_num, task_name))
    mod_obj = {}
    for mat in mat_opts:
        name = '{}_{}_{}_{}'.format(project, min_num, task_name, mat)
        print(name)
        cur_model = get_map_model(results_path, 'model_{}'.format(name))

        cur_data = joblib.load(data_path / 'data_{}.pkl'.format(name))

        assert len(cur_model.z) == len(cur_data['SI']), 'Number of participants in data and model do not match'
        num_patients = len(cur_model.z)

        err_arr = np.zeros((num_patients, 1))
        for i in range(0, num_patients):
            y_real, y_pred_mean = calc_y_pred_spec_data(full_data, cur_data, i, slope=False, model_box=cur_model)
            err_arr[i] = calc_error(y_real, y_pred_mean)

        num_clust = len(np.where(cur_model.allocmodel.Nk > 0)[0])
        seed = cur_model.rand_seed
        mod_type = cur_model.kernel
        best_ll = cur_model.best_ll
        si = cur_data['SI']

        seed_name = '{}_seed_{}'.format(name, seed)
        print(seed_name)
        mod_obj[seed_name] = ModelSum(task_name, mat, seed, mod_type, err_arr, num_clust, best_ll, si)
    return mod_obj


def calc_mogp_rmse_slope(data_path, project, min_num, task_name, mat_opts):
    """Calcluate rmse error between witheld test data and a linear slope fit to training data"""
    full_data = joblib.load(data_path / 'data_{}_{}_{}_full.pkl'.format(project, min_num, task_name))
    mod_obj = {}
    for mat in mat_opts:
        name = "{}_{}_{}_{}".format(project, min_num, task_name, mat)
        print(name)

        cur_data = joblib.load(data_path / 'data_{}.pkl'.format(name))

        num_patients = len(cur_data['SI'])
        err_arr = np.zeros((num_patients, 1))
        for i in range(0, num_patients):
            y_real, y_pred_mean = calc_y_pred_spec_data(full_data, cur_data, i, slope=True)
            err_arr[i] = calc_error(y_real, y_pred_mean)

        num_clust = num_patients
        seed = np.nan
        mod_type = 'slope'
        best_ll = np.nan
        si = cur_data['SI']

        mod_obj[name] = ModelSum(task_name, mat, seed, mod_type, err_arr, num_clust, best_ll, si)
    return mod_obj


def gen_mod_obj_full(project, task_name, save=False):
    """Generate model objects for sparsity and prediction experiments"""
    exp_path = Path('data/model_data/2_sparsity_prediction')
    if task_name == 'predict':
        min_num = 'min4'
        mat_opts = ['0.25', '0.5', '1.0', '1.5', '2.0']
        data_path = exp_path / 'prediction'

    elif task_name == 'sparse':
        min_num = 'min10'
        mat_opts = ['25', '50', '75']
        data_path = exp_path / 'sparsity'

    res_path_rbf = data_path / 'results' / 'rbf'
    res_path_lin = data_path / 'results' / 'linear'
    mod_obj_lin = calc_mogp_rmse(data_path, res_path_lin, project, min_num, task_name, mat_opts)
    mod_obj_rbf = calc_mogp_rmse(data_path, res_path_rbf, project, min_num, task_name, mat_opts)
    mod_obj_slope = calc_mogp_rmse_slope(data_path, project, min_num, task_name, mat_opts)

    if save:
        save_path = data_path / 'results' / 'rmse'
        Path.mkdir(save_path, parents=True, exist_ok=True)
        print(save_path.resolve())
        joblib.dump(mod_obj_lin, save_path / '{}_{}_linear_kernel_rmse_err.pkl'.format(task_name, project))
        joblib.dump(mod_obj_rbf, save_path / '{}_{}_rbf_kernel_rmse_err.pkl'.format(task_name, project))
        joblib.dump(mod_obj_slope, save_path / '{}_{}_slope_rmse_err.pkl'.format(task_name, project))

    return mod_obj_lin, mod_obj_rbf, mod_obj_slope


if __name__ == "__main__":
    # calculates RMSE error for prediction/sparsity experiments between witheld data and trajectory mean function
    _, _, _ = gen_mod_obj_full('ceft', 'sparse', save=True)
    _, _, _ = gen_mod_obj_full('ceft', 'predict', save=True)
    _, _, _ = gen_mod_obj_full('proact', 'sparse', save=True)
    _, _, _ = gen_mod_obj_full('proact', 'predict', save=True)
