#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


def shift_onset(name, group, df_onset, visit_col='Visit_Date', onset_col='onsetdt'):
    """Calculate time since symptom onset in years by subtracting visit date from date of symptom onset"""
    group[visit_col] = (group[visit_col] - df_onset.loc[name][onset_col]) / 365.4
    return group


def shift_onset_gtac(group):
    """For GTAC, calculate time since symptom onset in years"""
    group['Visit_Date'] = ((group['days since enrollment'] / 365.4) + (group['Duration at Enrollment Months']) / 12)
    return group


def shift_onset_cef(name, group, df_stat):
    """For Ceftriaxone, subtract age at symptom onset from age at visit date to get time in years from symptom onset"""

    group['Visit_Date'] = (group['age_at_DATE_PERFORMED'] - df_stat.loc[name]['age_at_ALS_DATE_SYM'])
    return group


def clean_aals(onset_file, alsfrs_file, column_labels):
    """ Generate consistent pandas dataframe for timeseries ALSFRS-R scores from AnswerALS files

    Arguments:
        onset_file (Path): path to AnswerALS file that includes age at ALS symptom onset ('v_NB_IATI_AALSHXFX.csv')
        alsfrs_file (Path): path to AnswerALS file that includes ALSFRS-R scores ('v_NB_IATI_ALSFRS_R.csv')
        column_labels (list): list of column names in final matrix
    Returns:
        df_aals_time_sub (pd.DataFrame): timeseries clinical measurements, in years from symptom onset
     """

    # Load files into pandas dataframes and convert to numeric values
    df_aals_onset = pd.read_csv(onset_file).set_index('SubjectUID').apply(pd.to_numeric, errors='coerce')
    df_aals_time = pd.read_csv(alsfrs_file).set_index('SubjectUID').apply(pd.to_numeric, errors='coerce')

    # Replace "Visit_Date" with "alsfrstdt" column to indicate date of alsfrs-r measurement
    df_aals_time.dropna(subset=['alsfrsdt'], inplace=True)
    df_aals_time['Visit_Date'] = df_aals_time['alsfrsdt']

    # Shift dataframe by date of symptom onset
    onset_list = [x for x in df_aals_onset['onsetdt'].dropna().index if x in df_aals_time.index]
    df_aals_time = df_aals_time.loc[onset_list]
    df_aals_time = df_aals_time.reset_index().groupby('SubjectUID').apply(
        lambda x: shift_onset(x.name, x, df_aals_onset, visit_col='Visit_Date', onset_col='onsetdt'))

    df_aals_time['dataset'] = 'aals'
    df_aals_time_sub = df_aals_time[column_labels]

    return df_aals_time_sub


def clean_proact_time(onset_file, clin_file, column_labels):
    """ Generate consistent pandas dataframe for timeseries scores from PROACT files - for ALSFRS-R and FVCP

    Arguments:
        onset_file (Path): path to PROACT file that includes age at ALS symptom onset ('AlsHistory.csv')
        clin_file (Path): path to PROACT file that includes timeseries clinical scores (ALSFRS-R: 'alsfrs.csv' or FVC: 'Fvc.csv')
        column_labels (list): list of column names in final matrix (alsfrs-r or fvc columns)
    Returns:
        df_proact_time_sub (pd.DataFrame): timeseries clinical measurements, in years from symptom onset
     """

    # Load files into pandas dataframes and convert to numeric values
    df_proact_onset = pd.read_csv(onset_file).set_index('subject_id').apply(pd.to_numeric, errors='coerce')
    df_proact_time = pd.read_csv(clin_file).set_index('subject_id').apply(pd.to_numeric, errors='coerce')

    seldel = [col for col in df_proact_time.columns if 'Delta' in col][0]
    df_proact_time.rename(columns={seldel: 'Delta'}, inplace=True)
    df_proact_time.dropna(subset=['Delta'], inplace=True)
    df_proact_onset = df_proact_onset[['Onset_Delta']].reset_index().drop_duplicates().dropna().set_index('subject_id')

    # Shift dataframe by date of symptom onset
    onset_list = [x for x in df_proact_onset['Onset_Delta'].dropna().index if x in df_proact_time.index]
    df_proact_time = df_proact_time.loc[onset_list]
    df_proact_time = df_proact_time.reset_index().groupby('subject_id').apply(
        lambda x: shift_onset(x.name, x, df_proact_onset, visit_col='Delta', onset_col='Onset_Delta'))

    # Harmonize column names
    pro_coldict = {'subject_id': 'SubjectUID', 'Delta': 'Visit_Date', 'Q1_Speech': 'alsfrs1',
                   'Q2_Salivation': 'alsfrs2', 'Q3_Swallowing': 'alsfrs3', 'Q4_Handwriting': 'alsfrs4',
                   'Q5a_Cutting_without_Gastrostomy': 'alsfrs5a', 'Q5b_Cutting_with_Gastrostomy': 'alsfrs5b',
                   'Q6_Dressing_and_Hygiene': 'alsfrs6', 'Q7_Turning_in_Bed': 'alsfrs7',
                   'Q8_Walking': 'alsfrs8', 'Q9_Climbing_Stairs': 'alsfrs9', 'R_1_Dyspnea': 'alsfrsr1',
                   'R_2_Orthopnea': 'alsfrsr2', 'R_3_Respiratory_Insufficiency': 'alsfrsr3',
                   'ALSFRS_R_Total': 'alsfrst',
                   'pct_of_Normal_Trial_1': 'fvcp1', 'pct_of_Normal_Trial_2': 'fvcp2', 'pct_of_Normal_Trial_3': 'fvcp3',
                   }

    df_proact_time.rename(columns=pro_coldict, inplace=True)

    df_proact_time['dataset'] = 'proact'
    df_proact_time_sub = df_proact_time[column_labels]

    return df_proact_time_sub


def clean_proact_survival(onset_file, death_file, df_time):
    """ Clean and format PROACT survival data for Kaplan-Meier analysis

    Arguments:
        onset_file (Path): path to PROACT file that includes age at ALS symptom onset ('AlsHistory.csv')
        death_file (Path): path to PROACT file that includes death data ('
        df_time (pd.DataFrame): timeseries clinical measurements, in years from symptom onset ('DeathData.csv')
    Returns:
        df_death_onset (pd.DataFrame): dataframe with disease duration and death event (boolean value indicating participant death)

     """

    df_proact_onset = pd.read_csv(onset_file).set_index('subject_id').apply(pd.to_numeric, errors='coerce')
    df_proact_death = pd.read_csv(death_file).set_index('subject_id')

    df_proact_death['Death_Days'] = pd.to_numeric(df_proact_death['Death_Days'], errors='coerce')

    # Clean duplicate values
    df_proact_onset = df_proact_onset[['Onset_Delta']].reset_index().drop_duplicates().dropna().set_index('subject_id')
    df_proact_death = df_proact_death[~df_proact_death.index.duplicated()]

    # Calculate time of last clinical visit measurement (proxy for last known date alive)
    df_max_visit = (df_time.reset_index().groupby('SubjectUID')['Visit_Date'].max())
    df_max_visit.name = 'Max_Delta'
    df_death_onset = df_proact_onset.join([df_proact_death, df_max_visit])

    # If participant has died, calculate survival duration
    df_death_onset_yes = df_death_onset[df_death_onset['Subject_Died'] == 'Yes'].copy().dropna(
        subset=['Death_Days', 'Onset_Delta'])
    df_death_onset_yes['disease_duration'] = (df_death_onset_yes['Death_Days'] - df_death_onset_yes[
        'Onset_Delta']) / 365.4

    # If participant has not died, calculate disease duration date using last known date alive
    df_death_onset_no = df_death_onset[df_death_onset['Subject_Died'] != 'Yes'].copy().dropna(
        subset=['Max_Delta', 'Onset_Delta'])
    df_death_onset_no['disease_duration'] = df_death_onset_no['Max_Delta']
    df_death_onset_no['Subject_Died'] = 'No'

    df_death_onset = df_death_onset_yes.append(df_death_onset_no)
    df_death_onset = df_death_onset.replace({'Subject_Died': {'Yes': True, 'No': False}})
    df_death_onset = df_death_onset.rename(columns={'Subject_Died': 'death_event'})
    df_death_onset.reset_index(inplace=True)
    df_death_onset['subj_proj_id'] = df_death_onset['subject_id'].astype('str') + '_' + 'proact'
    df_death_onset = df_death_onset[['subj_proj_id', 'subject_id', 'disease_duration', 'death_event']]
    return df_death_onset


def clean_emory(emory_file, column_labels):
    """ Generate consistent pandas dataframe for timeseries ALSFRS-R scores from Emory files

    Arguments:
        emory_file (Path): path to emory aggregated clinical file
        column_labels (list): list of column names in final matrix
    Returns:
        df_emory_time_sub (pd.DataFrame): timeseries clinical measurements, in years from symptom onset
     """
    # Load Data, Harmonize Columns
    df_emory_time = pd.read_excel(emory_file)
    emory_coldict = {'Patient ID': 'SubjectUID', 'Gender': 'sex', 'Age Onset': 'age_onset',
                     'C9orf72 testing': 'c9orf72', 'FRS Days from Onset': 'Visit_Date', '1 speech': 'alsfrs1',
                     '2 salivations': 'alsfrs2', '3 swallow': 'alsfrs3', '4 handwrit': 'alsfrs4',
                     '5 wo gastr.': 'alsfrs5a', '5 w gastr.': 'alsfrs5b',
                     '6 dress hygn': 'alsfrs6', '7 turnbed': 'alsfrs7', '8 walking': 'alsfrs8', '9 clmbstrs': 'alsfrs9',
                     '10 dyspnea': 'alsfrsr1', '11 orthpnea': 'alsfrsr2', '12 resp insf': 'alsfrsr3',
                     'FRSRTotal': 'alsfrst'}
    df_emory_time.rename(columns=emory_coldict, inplace=True)
    df_emory_time['dataset'] = 'emory'

    df_emory_time['Visit_Date'] = df_emory_time['Visit_Date'] / 365.4

    df_emory_time_sub = df_emory_time[column_labels]

    return df_emory_time_sub


def clean_gtac(gtac_file, column_labels):
    """ Generate consistent pandas dataframe for timeseries ALSFRS-R scores from GTAC files

    Arguments:
        gtac_file (Path): path to aggregated gtac clinical data file, including disease duration at enrollment, alsfrs-r scores
        column_labels (list): list of column names in final matrix
    Returns:
        df_gtac_time_sub (pd.DataFrame): timeseries clinical measurements, in years from symptom onset
     """
    # Load data, shift visit dates by symptom onset
    df_gtac = pd.read_excel(gtac_file).set_index('NeuroGuid').apply(pd.to_numeric, errors='coerce')
    df_gtac_shift = df_gtac.dropna(subset=['Duration at Enrollment Months'])
    df_gtac_shift = df_gtac_shift.reset_index().groupby('NeuroGuid').apply(lambda x: shift_onset_gtac(x))

    # Harmonize column names
    gtac_coldict = {'NeuroGuid': 'SubjectUID', 'Age at Onset': 'age_onset', 'Sex': 'sex', 'ALSFRSR Total': 'alsfrst'}
    df_gtac_shift.rename(columns=gtac_coldict, inplace=True)
    df_gtac_shift['dataset'] = 'gtac'

    df_gtac_time_sub = df_gtac_shift[column_labels]

    return df_gtac_time_sub


def clean_ceft(onset_file, alsfrs_file, column_labels):
    """ Generate consistent pandas dataframe for timeseries alsfrs-r scores from Ceftriaxone files

    Arguments:
        onset_file (Path): path to ceftriaxone file that includes age at ALS symptom onset ('mhas.csv')
        alsfrs_file (Path): path to ceftriaxone alsfrsr-r file ('alsf.csv')
        column_labels (list): list of column names in final matrix
    Returns:
        df_cef_time_sub (pd.DataFrame): timeseries clinical measurements, in years from symptom onset
     """

    # Load files into pandas dataframes
    df_cef_onset = pd.read_csv(onset_file).set_index('study_id')
    df_cef_time = pd.read_csv(alsfrs_file).set_index('study_id')

    # Shift dataframe by date of symptom onset
    df_cef_time = df_cef_time.loc[df_cef_onset['age_at_ALS_DATE_SYM'].dropna().index]
    df_cef_time = df_cef_time.reset_index().groupby('study_id').apply(
        lambda x: shift_onset_cef(x.name, x, df_cef_onset))

    # Harmonize column names
    cef_coldict = {'study_id': 'SubjectUID', 'speech': 'alsfrs1',
                   'salivation': 'alsfrs2', 'swallowing': 'alsfrs3', 'handwriting': 'alsfrs4',
                   'cutting_with': 'alsfrs5a', 'cutting_without': 'alsfrs5b',
                   'dressing': 'alsfrs6', 'turning': 'alsfrs7', 'walking': 'alsfrs8', 'climbing': 'alsfrs9',
                   'dyspnea': 'alsfrsr1', 'orthopnea': 'alsfrsr2', 'respiratory': 'alsfrsr3', 'als_total': 'alsfrst'}
    df_cef_time.rename(columns=cef_coldict, inplace=True)

    df_cef_time['dataset'] = 'ceft'
    df_cef_time_sub = df_cef_time[column_labels]

    return df_cef_time_sub

def clean_nathist(onset_file, alsfrs_file, column_labels):
    """ Generate consistent pandas dataframe for timeseries ALSFRS-R scores from ALS Natural History files

    Arguments:
        onset_file (Path): path to AnswerALS file that includes age at ALS symptom onset ('v_NB_CLIN_NALSHXFX.csv')
        alsfrs_file (Path): path to AnswerALS file that includes ALSFRS-R scores ('v_NB_CLIN_ALSFRS_R.csv')
        column_labels (list): list of column names in final matrix
    Returns:
        df_aals_time_sub (pd.DataFrame): timeseries clinical measurements, in years from symptom onset
     """

    # Load files into pandas dataframes and convert to numeric values
    df_nathist_onset = pd.read_csv(onset_file)
    df_nathist_onset.rename(columns={'Neurostamp':'SubjectUID'}, inplace=True)
    df_nathist_onset = df_nathist_onset.set_index('SubjectUID').apply(pd.to_numeric, errors='coerce')
    df_nathist_time = pd.read_csv(alsfrs_file)
    df_nathist_time.rename(columns={'Neurostamp': 'SubjectUID'}, inplace=True)
    df_nathist_time = df_nathist_time.set_index('SubjectUID').apply(pd.to_numeric, errors='coerce')

    # Replace "Visit_Date" with "alsfrstdt" column to indicate date of alsfrs-r measurement
    df_nathist_time.dropna(subset=['alsfrsdt'], inplace=True)
    df_nathist_time['Visit_Date'] = df_nathist_time['alsfrsdt']

    # Shift dataframe by date of symptom onset
    onset_list = [x for x in df_nathist_onset['onsetdt'].dropna().index if x in df_nathist_time.index]
    df_nathist_time = df_nathist_time.loc[onset_list]
    # df_nathist_time = df_nathist_time.reset_index().groupby('SubjectUID').apply(
    #     lambda x: shift_onset(x.name, x, df_nathist_onset, visit_col='Visit_Date', onset_col='onsetdt'))
    df_nathist_time = df_nathist_time.join(df_nathist_onset[['onsetdt']], how='left')
    df_nathist_time['Visit_Date'] = (df_nathist_time['Visit_Date'] - df_nathist_time['onsetdt']) / 365.4
    df_nathist_time.reset_index(inplace=True)

    df_nathist_time['dataset'] = 'nathist'
    df_nathist_time_sub = df_nathist_time[column_labels]

    return df_nathist_time_sub

def calc_clean_summary_stats(df_time, category):
    """Calculate summary statistics (ALSFRS-R subscores, vital capacity max/average)"""

    # Add subject-project identifier, drop incomplete rows
    df_time['subj_proj_id'] = df_time['SubjectUID'].astype('str') + '_' + df_time['dataset']

    if category == 'alsfrs':
        df_time['alsfrst_bulb'] = (df_time[['alsfrs1', 'alsfrs2', 'alsfrs3']]).sum(axis=1, skipna=False)
        df_time['alsfrst_fine'] = (df_time[['alsfrs4', 'alsfrs5a', 'alsfrs5b', 'alsfrs6']]).sum(axis=1, min_count=3)
        df_time['alsfrst_gross'] = (df_time[['alsfrs7', 'alsfrs8', 'alsfrs9']]).sum(axis=1, skipna=False)
        df_time['alsfrst_resp'] = (df_time[['alsfrsr1', 'alsfrsr2', 'alsfrsr3']]).sum(axis=1, skipna=False)

        df_time = df_time[
            ['subj_proj_id', 'SubjectUID', 'dataset', 'Visit_Date', 'alsfrst_bulb', 'alsfrst_fine', 'alsfrst_gross',
             'alsfrst_resp', 'alsfrst']].drop_duplicates().dropna(how='any')

        # Drop ALSFRS_R rows if subscores do not sum correctly to total
        df_time = df_time[(df_time['alsfrst']==(df_time[['alsfrst_bulb', 'alsfrst_fine', 'alsfrst_gross', 'alsfrst_resp']]).sum(axis=1))].copy()

    elif category == 'fvc':
        df_time['fvcp_avg'] = (df_time[['fvcp1', 'fvcp2', 'fvcp3']]).mean(axis=1)
        df_time['fvcp_max'] = (df_time[['fvcp1', 'fvcp2', 'fvcp3']]).max(axis=1)

        df_time = df_time[
            ['subj_proj_id', 'SubjectUID', 'dataset', 'Visit_Date', 'fvcp_avg', 'fvcp_max']].drop_duplicates().dropna(
            how='any')

    elif category == 'svc':
        df_time['svcp_avg'] = (df_time[['svcp1', 'svcp2', 'svcp3']]).mean(axis=1)
        df_time['svcp_max'] = (df_time[['svcp1', 'svcp2', 'svcp3']]).max(axis=1)

        df_time = df_time[
            ['subj_proj_id', 'SubjectUID', 'dataset', 'Visit_Date', 'svcp_avg', 'svcp_max']].drop_duplicates().dropna(
            how='any')

    # Remove any patients with Visit Dates less than 0
    # Affects 5 participants in ALSFRS-R: Emory may be data entry error
    df_time = df_time[~df_time['SubjectUID'].isin(df_time[df_time['Visit_Date'] < 0]['SubjectUID'])].copy()

    # If participant has multiple score measurements on a single day, take average
    df_time = df_time.groupby(['subj_proj_id', 'SubjectUID', 'dataset', 'Visit_Date']).mean().reset_index()
    df_time = df_time.sort_values(by=['dataset', 'SubjectUID', 'Visit_Date'])

    return df_time


if __name__ == "__main__":
    alsfrs_cats = ['SubjectUID', 'Visit_Date', 'alsfrs1', 'alsfrs2', 'alsfrs3', 'alsfrs4',
                   'alsfrs5a', 'alsfrs5b', 'alsfrs6', 'alsfrs7', 'alsfrs8', 'alsfrs9',
                   'alsfrsr1', 'alsfrsr2', 'alsfrsr3', 'alsfrst', 'dataset']

    fvcp_cats = ['SubjectUID', 'Visit_Date', 'fvcp1', 'fvcp2', 'fvcp3', 'dataset']

    # Set data paths
    emory_file_path = Path('data/raw_data/emory/emory_deidentified_04012020.xlsx')
    assert emory_file_path.exists(), 'missing emory file'

    gtac_file_path = Path('data/raw_data/gtac/gtac_deidentified_03122020.xlsx')
    assert gtac_file_path.exists(), 'missing gtac file'

    ceft_data_path = Path('data/processed_data/ceft')
    ceft_data_onset = ceft_data_path / 'mhas.csv'
    ceft_data_alsfrs = ceft_data_path / 'alsf.csv'
    assert (ceft_data_onset.exists() & ceft_data_alsfrs.exists()), 'missing ceft file'

    aals_data_path = Path('data/raw_data/aals')
    aals_data_onset = aals_data_path / 'v_NB_IATI_AALSHXFX.csv'
    aals_data_alsfrs = aals_data_path / 'v_NB_IATI_ALSFRS_R.csv'
    assert (aals_data_onset.exists() & aals_data_alsfrs.exists()), 'missing aals file'

    proact_data_path = Path('data/raw_data/proact')
    proact_data_onset = proact_data_path / 'AlsHistory.csv'
    proact_data_alsfrs = proact_data_path / 'alsfrs.csv'
    proact_data_fvc = proact_data_path / 'Fvc.csv'
    proact_data_survival = proact_data_path / 'DeathData.csv'
    assert (proact_data_onset.exists() & proact_data_alsfrs.exists()
            & proact_data_fvc.exists() & proact_data_survival.exists()), 'missing proact file'

    nathist_data_path = Path('data/raw_data/nathist')
    nathist_data_onset = nathist_data_path / 'v_NB_CLIN_NALSHXFX.csv'
    nathist_data_alsfrs = nathist_data_path / 'v_NB_CLIN_ALSFRS_R.csv'
    assert (nathist_data_onset.exists() & nathist_data_alsfrs.exists()), 'missing nathist file'

    # Load and clean dataframes
    df_time_emory = clean_emory(emory_file_path, alsfrs_cats)
    df_time_gtac = clean_gtac(gtac_file_path, alsfrs_cats)
    df_time_ceft = clean_ceft(ceft_data_onset, ceft_data_alsfrs, alsfrs_cats)
    df_time_aals = clean_aals(aals_data_onset, aals_data_alsfrs, alsfrs_cats)
    df_time_nathist = clean_nathist(nathist_data_onset, nathist_data_alsfrs, alsfrs_cats)

    df_time_proact_alsfrs = clean_proact_time(proact_data_onset, proact_data_alsfrs, alsfrs_cats)
    df_time_proact_fvc = clean_proact_time(proact_data_onset, proact_data_fvc, fvcp_cats)
    df_stat_proact_survival = clean_proact_survival(proact_data_onset, proact_data_survival, df_time_proact_alsfrs)

    df_time_merge_alsfrs = df_time_aals.append([df_time_proact_alsfrs, df_time_emory, df_time_gtac, df_time_ceft, df_time_nathist])
    df_time_merge_alsfrs = calc_clean_summary_stats(df_time_merge_alsfrs, 'alsfrs')
    df_time_proact_fvc = calc_clean_summary_stats(df_time_proact_fvc, 'fvc')

    # # Save dataframes
    df_time_merge_alsfrs.to_csv('data/processed_data/timeseries_all_alsfrsr.csv', index=False)
    
    df_time_proact_fvc.to_csv('data/processed_data/timeseries_proact_fvcp.csv', index=False)
    df_stat_proact_survival.to_csv('data/processed_data/static_proact_death.csv', index=False)
