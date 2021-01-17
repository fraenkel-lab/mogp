import math
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


def incl_crit(data, subj_col, delta_col, min_data=3, jump_size=6, jump_col='alsfrst', max_year_onset=7):
    """Filter pandas dataframe by inclusion criteria"""
    data['Visit_Number'] = data.groupby(subj_col)[delta_col].rank()
    visit_ic = (data.groupby(subj_col)['Visit_Number'].max() >= min_data)
    onset_ic = (data.groupby(subj_col)[delta_col].min() < max_year_onset)
    if jump_size is None:
        data = data.set_index(subj_col).loc[visit_ic & onset_ic].reset_index()
    else:
        jump_ic = data.groupby([subj_col])[jump_col].apply(lambda x: ~(np.diff(x) > jump_size).any())
        data = data.set_index(subj_col).loc[visit_ic & onset_ic & jump_ic].reset_index()
    return data


def process_data_mogp_pre_norm(data, curfeat, subj_col, delta_col, max_feat=48.):
    """Convert pandas dataframe to data dictionary for MoGP input, including adding onset anchor datapoint"""
    data_dict = {}

    X_df = data.pivot(index=subj_col, columns='Visit_Number', values=delta_col)
    Y_df = data.pivot(index=subj_col, columns='Visit_Number', values=curfeat)
    assert (X_df.index == Y_df.index).all(), 'X and Y df indexes do not match'

    SI = list(X_df.index)
    Y_mean = Y_df.stack().mean()
    Y_std = Y_df.stack().std()

    # Add onset anchor value at symptom onset
    X_df.insert(0, 0.0, 0.)
    Y_df.insert(0, 0.0, max_feat * np.ones(data[subj_col].nunique()))

    data_dict['SI'] = SI
    data_dict['XA'] = X_df.to_numpy()
    data_dict['YA'] = Y_df.to_numpy()
    data_dict['Y_mean'] = Y_mean
    data_dict['Y_std'] = Y_std
    return data_dict


def process_data_mogp_pandas(data, curfeat, subj_col, delta_col, max_feat=48., min_data=3, jump_size=6, max_year_onset=7):
    """Filter data with inclusion criteria, convert to data dictionary, and z-score normalize"""
    data = incl_crit(data=data, subj_col=subj_col, delta_col=delta_col, min_data=min_data, jump_size=jump_size, max_year_onset=max_year_onset)
    data_dict = process_data_mogp_pre_norm(data=data, curfeat=curfeat, subj_col=subj_col, delta_col=delta_col, max_feat=max_feat)
    return data_dict


def gen_predict_mogp_data(cur_dict, pred_time):
    """Generate prediction experiments data dictionary"""
    X = cur_dict['XA']
    Y = cur_dict['YA']
    SI = cur_dict['SI']
    n = len(SI)

    XA = np.nan * np.ones(X.shape)
    YA = np.nan * np.ones(Y.shape)
    SIA = []
    data_dict = {}
    ctr = 0
    for i in np.arange(n):
        first_visit_date = X[i, 1]
        max_visit_date = first_visit_date + pred_time
        x_sel = [x for x in X[i, :] if x <= max_visit_date]
        y_sel = Y[i, :len(x_sel)]
        XA[ctr, :len(x_sel)] = x_sel
        YA[ctr, :len(x_sel)] = y_sel
        SIA.append(SI[i])
        ctr += 1

    data_dict['SI'] = SIA
    data_dict['XA'] = XA
    data_dict['YA'] = YA
    return data_dict


def gen_sparse_mogp_data(cur_dict, perc, rand_seed=0):
    """Generate sparsity experiments data dictionary"""
    np.random.seed(rand_seed)

    X = cur_dict['XA']
    Y = cur_dict['YA']
    SI = cur_dict['SI']
    n = len(SI)

    XA = np.nan * np.ones(X.shape)
    YA = np.nan * np.ones(Y.shape)
    SIA = []
    data_dict = {}
    ctr = 0
    for i in np.arange(n):
        idx = ~np.isnan(X[i, :])

        # select percentage of real data (total num data - 1, because 1 anchor onset)
        sum_idx = np.sum(idx)
        rand_sel = np.sort(np.random.choice(np.arange(1, sum_idx), math.floor((sum_idx - 1) * perc), replace=False))
        rand_sel = np.append(0, rand_sel)  # always include onset anchor

        XA[ctr, :len(rand_sel)] = X[i, rand_sel]
        YA[ctr, :len(rand_sel)] = Y[i, rand_sel]
        SIA.append(SI[i])
        ctr += 1

    data_dict['SI'] = SIA
    data_dict['XA'] = XA
    data_dict['YA'] = YA
    return data_dict


def split_train_test(data, project, train_size=0.6, random_seed=0):
    """Split test and training data for reference model experiments"""
    np.random.seed(random_seed)
    data_proj = data[data['dataset'] == project]
    pat_lis = data_proj['subj_proj_id'].unique()
    train_pats = np.random.choice(pat_lis, round(len(pat_lis)*train_size), replace=False)
    test_pats = list(set(pat_lis)-set(train_pats))
    # print('{}: Num pats in train: {}, test: {}'.format(project, len(train_pats), len(test_pats)))
    train_data = data_proj[data_proj['subj_proj_id'].isin(train_pats)]
    test_data = data_proj[data_proj['subj_proj_id'].isin(test_pats)]
    return train_data, test_data

###########################################################################
#######                Process Data for Experiments                 #######
###########################################################################


def data_full_alsfrst(df_time_merge, exp_path):
    """Experiment: Full ALSFRS-R"""
    Path.mkdir(exp_path, parents=True, exist_ok=True)

    for proj in ['aals', 'proact', 'gtac', 'emory', 'ceft']:
        df_time_proj = df_time_merge[df_time_merge['dataset'] == proj].copy()
        proj_dict = process_data_mogp_pandas(df_time_proj, 'alsfrst', 'subj_proj_id', 'Visit_Date', max_feat=48., min_data=3, jump_size=6, max_year_onset=7)
        joblib.dump(proj_dict, exp_path / 'data_{}_min3_alsfrst.pkl'.format(proj))


def data_sparse(df_time_merge, exp_path):
    """Experiment: Interpolation"""
    Path.mkdir(exp_path, parents=True, exist_ok=True)
    proj_list = ['ceft', 'proact']
    proj_min_visits = 10
    sparse_list = [25, 50, 75]

    for proj in proj_list:
        # Generate full dictionary
        sparse_dict_pandas = process_data_mogp_pandas(df_time_merge[df_time_merge['dataset'] == proj].copy(),
                                                      'alsfrst', 'SubjectUID', 'Visit_Date', min_data=proj_min_visits, jump_size=6, max_year_onset=7)
        joblib.dump(sparse_dict_pandas, exp_path / 'data_{}_min{}_sparse_full.pkl'.format(proj, proj_min_visits))

        # Generate sparse dictionaries
        for sparse_time in sparse_list:
            cur_sparse_mat = gen_sparse_mogp_data(sparse_dict_pandas, sparse_time / 100)
            joblib.dump(cur_sparse_mat, exp_path / 'data_{}_min{}_sparse_{}.pkl'.format(proj, proj_min_visits, sparse_time))


def data_predict(df_time_merge, exp_path):
    """Experiment: Prediction"""
    Path.mkdir(exp_path, parents=True, exist_ok=True)
    proj_list = ['ceft', 'proact']
    proj_min_visits = 4
    pred_lis = [0.25, 0.50, 1.0, 1.5, 2.0]

    for proj in proj_list:
        pred_dict_pandas = process_data_mogp_pandas(df_time_merge[df_time_merge['dataset'] == proj].copy(),
                                                    'alsfrst', 'SubjectUID', 'Visit_Date', min_data=proj_min_visits, jump_size=6, max_year_onset=7)
        joblib.dump(pred_dict_pandas,  exp_path / 'data_{}_min{}_predict_full.pkl'.format(proj, proj_min_visits))

        for pred_time in pred_lis:
            cur_pred_mat = gen_predict_mogp_data(pred_dict_pandas, pred_time)
            joblib.dump(cur_pred_mat, exp_path / 'data_{}_min{}_predict_{}.pkl'.format(proj, proj_min_visits, pred_time))


def data_reference_by_split(data_ic, exp_path, random_seed=0):
    """Experiment: Reference - for individual split"""
    # reference model - proact train
    df_time_proact_train, df_time_proact_test = split_train_test(data_ic, 'proact', train_size=0.6, random_seed=random_seed)
    pro_train_dict = process_data_mogp_pre_norm(df_time_proact_train.copy(), 'alsfrst', 'subj_proj_id', 'Visit_Date', max_feat=48.)
    joblib.dump(pro_train_dict, exp_path / 'data_proact_min3_alsfrst_train_split_{}.pkl'.format(random_seed))

    # proact - test
    proact_test_dict = process_data_mogp_pre_norm(df_time_proact_test.copy(), 'alsfrst', 'subj_proj_id', 'Visit_Date', max_feat=48.)
    joblib.dump(proact_test_dict, exp_path / 'data_proact_min3_alsfrst_test_split_{}.pkl'.format(random_seed))

    # all other data - test and train
    for proj in ['aals', 'gtac', 'emory', 'ceft']:
        df_time_proj_train, df_time_proj_test = split_train_test(data_ic, proj, train_size=0.6, random_seed=random_seed)

        proj_train_dict = process_data_mogp_pre_norm(df_time_proj_train.copy(), 'alsfrst', 'subj_proj_id', 'Visit_Date', max_feat=48.)
        joblib.dump(proj_train_dict, exp_path / 'data_{}_min3_alsfrst_train_split_{}.pkl'.format(proj, random_seed))

        proj_test_dict = process_data_mogp_pre_norm(df_time_proj_test.copy(), 'alsfrst', 'subj_proj_id', 'Visit_Date', max_feat=48.)
        joblib.dump(proj_test_dict, exp_path / 'data_{}_min3_alsfrst_test_split_{}.pkl'.format(proj, random_seed))


def data_reference(df_time_merge, exp_path):
    """Experiment: Reference - all splits for repeated random sub-sampling validation"""
    Path.mkdir(exp_path, parents=True, exist_ok=True)
    num_splits = 5

    data_ic = incl_crit(df_time_merge.copy(), 'subj_proj_id', 'Visit_Date', min_data=3, jump_size=6, max_year_onset=7)
    for split in range(0, num_splits):
        data_reference_by_split(data_ic=data_ic, exp_path=exp_path, random_seed=split)


def data_alt_outcomes(df_time_merge, df_time_fvc, exp_path):
    """Experiment: Alternate Outcomes"""
    Path.mkdir(exp_path, parents=True, exist_ok=True)
    proj = 'proact'
    min_subscore_thresh = 11

    # PROACT - forced vital capacity (maximum value)
    proj_dict = process_data_mogp_pandas(df_time_fvc.copy(), 'fvcp_max', 'subj_proj_id', 'Visit_Date',
                                         max_feat=100., min_data=3, jump_size=None, max_year_onset=7)
    joblib.dump(proj_dict, exp_path / 'data_{}_min3_{}.pkl'.format(proj, 'fvcpmax'))

    # PROACT - ALSFRS-R Subscores
    df_time_proj = df_time_merge[df_time_merge['dataset'] == proj].copy()
    df_time_proact_alsfrsr = incl_crit(df_time_proj, 'subj_proj_id', 'Visit_Date', min_data=3, jump_size=6, max_year_onset=7)
    for cat in ['alsfrst_bulb', 'alsfrst_fine', 'alsfrst_gross', 'alsfrst_resp']:
        # Remove patients with minimally changing score (flat trajectories)
        pat_div = df_time_proact_alsfrsr.groupby('subj_proj_id')[cat].min() < min_subscore_thresh
        df_time_proact_alsfrsr_change = df_time_proact_alsfrsr[df_time_proact_alsfrsr['subj_proj_id'].isin(pat_div[pat_div].index)]
        df_time_proact_alsfrsr_nochange = df_time_proact_alsfrsr[~df_time_proact_alsfrsr['subj_proj_id'].isin(pat_div[pat_div].index)]

        cur_dict_pandas_change = process_data_mogp_pre_norm(df_time_proact_alsfrsr_change.copy(), cat, 'subj_proj_id', 'Visit_Date', max_feat=12.)
        cur_dict_pandas_nochange = process_data_mogp_pre_norm(df_time_proact_alsfrsr_nochange.copy(), cat, 'subj_proj_id', 'Visit_Date', max_feat=12.)

        # Save
        joblib.dump(cur_dict_pandas_change, exp_path / 'data_{}_min3_{}.pkl'.format(proj, cat))
        joblib.dump(cur_dict_pandas_nochange, exp_path / 'data_{}_min3_{}_nochange.pkl'.format(proj, cat))


if __name__ == '__main__':
    # Set intput/output paths
    timeseries_alsfrsr_path = Path('data/processed_data/timeseries_all_alsfrsr.csv')
    timeseries_proact_fvcp_path = Path('data/processed_data/timeseries_proact_fvcp.csv')

    full_alsfrst_path = Path('data/model_data/1_alsfrsr_all')
    sparse_path = Path('data/model_data/2_sparsity_prediction/sparsity')
    pred_path = Path('data/model_data/2_sparsity_prediction/prediction')
    ref_path = Path('data/model_data/3_reference_transfer')
    altclin_path = Path('data/model_data/4_proact_alt_endpoints')

    assert (timeseries_alsfrsr_path.exists() & timeseries_proact_fvcp_path.exists()), 'input files missing'

    # Process data files for experiments
    df_time_merge_alsfrs = pd.read_csv(timeseries_alsfrsr_path)
    df_time_proact_fvc = pd.read_csv(timeseries_proact_fvcp_path)

    data_full_alsfrst(df_time_merge_alsfrs, full_alsfrst_path)
    data_sparse(df_time_merge_alsfrs, sparse_path)
    data_predict(df_time_merge_alsfrs, pred_path)
    data_reference(df_time_merge_alsfrs, ref_path)
    data_alt_outcomes(df_time_merge_alsfrs, df_time_proact_fvc, altclin_path)
