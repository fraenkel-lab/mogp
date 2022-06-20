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


def process_data_mogp_pre_norm(data, curfeat, subj_col, delta_col, max_feat=48., anchor=True):
    """Convert pandas dataframe to data dictionary for MoGP input, including adding onset anchor datapoint"""
    data_dict = {}

    X_df = data.pivot(index=subj_col, columns='Visit_Number', values=delta_col)
    Y_df = data.pivot(index=subj_col, columns='Visit_Number', values=curfeat)
    assert (X_df.index == Y_df.index).all(), 'X and Y df indexes do not match'

    SI = list(X_df.index)
    Y_mean = Y_df.stack().mean()
    Y_std = Y_df.stack().std()

    # Add onset anchor value at symptom onset
    if anchor:
        X_df.insert(0, 0.0, 0.)
        Y_df.insert(0, 0.0, max_feat * np.ones(data[subj_col].nunique()))

    data_dict['SI'] = SI
    data_dict['XA'] = X_df.to_numpy()
    data_dict['YA'] = Y_df.to_numpy()
    data_dict['Y_mean'] = Y_mean
    data_dict['Y_std'] = Y_std
    return data_dict


def process_data_mogp_pandas(data, curfeat, subj_col, delta_col, max_feat=48., min_data=3, jump_size=6, max_year_onset=7, anchor=True):
    """Filter data with inclusion criteria, convert to data dictionary, and z-score normalize"""
    data = incl_crit(data=data, subj_col=subj_col, delta_col=delta_col, min_data=min_data, jump_size=jump_size, max_year_onset=max_year_onset)
    data_dict = process_data_mogp_pre_norm(data=data, curfeat=curfeat, subj_col=subj_col, delta_col=delta_col, max_feat=max_feat, anchor=anchor)
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
    proj_list = ['aals', 'proact', 'gtac', 'emory', 'ceft', 'nathist']
    for proj in proj_list:
        df_time_proj = df_time_merge[df_time_merge['dataset'] == proj].copy()
        proj_dict = process_data_mogp_pandas(df_time_proj, 'alsfrst', 'subj_proj_id', 'Visit_Date', max_feat=48., min_data=3, jump_size=6, max_year_onset=7)
        joblib.dump(proj_dict, exp_path / 'data_{}_min3_alsfrst.pkl'.format(proj))


def data_sparse(df_time_merge, exp_path, proj_min_visits=10):
    """Experiment: Interpolation"""
    Path.mkdir(exp_path, parents=True, exist_ok=True)
    proj_list = ['ceft', 'proact', 'aals', 'emory', 'nathist']
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


def data_predict(df_time_merge, exp_path, proj_min_visits=4):
    """Experiment: Prediction"""
    Path.mkdir(exp_path, parents=True, exist_ok=True)
    proj_list = ['ceft', 'proact', 'aals', 'emory', 'nathist']
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
    proj_list  = ['aals', 'gtac', 'emory', 'ceft', 'nathist']
    for proj in proj_list:
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

def data_adni(df_ad, df_diag, df_onset, exp_path):
    """Experiment: ADAS-Cog-13"""
    df_ad_sub = df_ad.dropna(subset=['ADAS13','Years_bl']).copy()

    # both minimum 3 visits and has AD diagnosis (exclude MCI/ HC)
    min3lis = df_ad_sub.groupby('PTID').size()[df_ad_sub.groupby('PTID').size()>=3].index
    ad_diaglis = df_diag[df_diag['DXCURREN']==3]['PTID']

    df_ad_min3 = df_ad_sub[df_ad_sub['PTID'].isin(set(min3lis)&set(ad_diaglis))].copy()
    df_ad_min3.dropna(subset=['Years_bl', 'PTID', 'ADAS13'], inplace=True)

    df_ad_min3.sort_values(by=['PTID','Years_bl'], inplace=True)
    df_ad_min3['Visit_Number']=df_ad_min3.groupby('PTID')['Years_bl'].rank("dense")

    df_ad_min3['neg-ADAS13']=-df_ad_min3['ADAS13']

    adni_dict = process_data_mogp_pre_norm(df_ad_min3, curfeat='neg-ADAS13', subj_col='PTID',  delta_col='Years_bl', anchor=False)

    joblib.dump(adni_dict, exp_path / 'data_adni_min3_adas13.pkl')

def data_ppmi(patient_status, mds, patient_diagnosis, exp_path):
    """Experiment: MDSD-UPDRS-PartIII-OffMed"""
    #Diagnosis history, SXDT = date of symptom, PDDXDT = date of Parkinson's disease diagnosis
    #Filter for PD cohort and OFF medication (note that NaN is prior to initation of medications,
    #All patients were required not to have started medication at enrollment)
    pd_ids = patient_status[(patient_status.Subgroup=='Sporadic') & (patient_status.Comments.isnull())].PATNO.unique()
    pd_mds = mds[np.isin(mds.PATNO, pd_ids) & (mds.PDSTATE != 'ON')]  #original inclusion
    updated_pd_mds = pd_mds[(pd_mds.PDMEDDT.isnull()) | (pd.notna(pd_mds.PDMEDDT) & (pd_mds.PDSTATE == 'OFF'))][['PATNO', 'EVENT_ID', 'INFODT', 'NP3TOT', 'PDSTATE']]
    pd_pd = patient_diagnosis[np.isin(patient_diagnosis.PATNO, pd_ids) & (patient_diagnosis.EVENT_ID == 'SC')]
    df = updated_pd_mds.merge(pd_pd[['PATNO', 'SXDT', 'PDDXDT']],  how='left', on='PATNO')

    # convert dates to usable numbers
    ix_as_num = []
    dx_as_num = []
    sx_as_num = []

    infodt = df.INFODT.str.split('/')
    sxdt = df.SXDT.str.split('/')
    pddt = df.PDDXDT.str.split('/')

    for i in range(len(df)):
        ix_as_num.append(float(infodt[i][0])/12. + float(infodt[i][1]))
        dx_as_num.append(float(pddt[i][0])/12. + float(pddt[i][1]))
        
        try:
            sx_as_num.append(float(sxdt[i][0])/12. + float(sxdt[i][1]))
        except:
            sx_as_num.append(np.NaN)
            
    df['Visit_date'] = ix_as_num
    df['Diagnosis_date'] = dx_as_num
    df['Symptom_date'] = sx_as_num

    df['Time_since_diagnosis'] = df['Visit_date'] - df['Diagnosis_date']
    df['Time_since_onset'] = df['Visit_date'] - df['Symptom_date']

    # sort by inclusion criteria
    min3_ppmi=df.groupby('PATNO').size()[df.groupby('PATNO').size()>=3].index
    onset_10y=df.groupby('PATNO')['Time_since_onset'].min()[df.groupby('PATNO')['Time_since_onset'].min()<10].index
    df_ppmi_min3 = df[df['PATNO'].isin(set(min3_ppmi)&set(onset_10y))].copy()
    df_ppmi_min3.dropna(subset=['Time_since_onset', 'PATNO', 'NP3TOT'], inplace=True)

    # # Clean duplicated visits
    df_ppmi_min3 = df_ppmi_min3.groupby(['PATNO', 'Time_since_onset']).mean().reset_index()
    df_ppmi_min3.sort_values(by=['PATNO', 'Time_since_onset'], inplace=True)
    df_ppmi_min3['Visit_Number']=df_ppmi_min3.groupby('PATNO')['Time_since_onset'].rank("dense")

    df_ppmi_min3['neg-NP3TOT']=-df_ppmi_min3['NP3TOT']

    ppmi_dict = process_data_mogp_pre_norm(df_ppmi_min3, curfeat='neg-NP3TOT', subj_col='PATNO',  delta_col='Time_since_onset', anchor=False)
    joblib.dump(ppmi_dict, exp_path / 'data_ppmi_min3_updrs.pkl')

if __name__ == '__main__':
    # # Set intput/output paths
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

    # input/outputs for ADNI/PPMI:
    nonals_path = Path('data/model_data/5_nonals_domains')
    adni_path = 'data/raw_data/adni'
    adni_ad = pd.read_csv(adni_path+'/Study Info/ADNIMERGE.csv')
    adni_diag = pd.read_csv(adni_path + '/Diagnosis/DXSUM_PDXCONV_ADNIALL.csv')
    adni_onset = pd.read_csv(adni_path + '/PTDEMOG.csv')
    data_adni(adni_ad, adni_diag, adni_onset, nonals_path)

    ppmi_path = 'data/raw_data/ppmi'
    ppmi_patient_status = pd.read_excel(ppmi_path + '/Consensus_Committee_Analytic_Datasets_28OCT21.xlsx', sheet_name='PD')
    ppmi_mds = pd.read_csv(ppmi_path + '/MDS_UPDRS_Part_III.csv')
    ppmi_patient_diagnosis = pd.read_csv(ppmi_path + '/PD_Diagnosis_History.csv')
    data_ppmi(ppmi_patient_status, ppmi_mds, ppmi_patient_diagnosis, nonals_path)

