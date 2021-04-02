# Experiment Workflow
This folder includes all scripts used to run experiments and generate manuscript figures.

MoGP can be computationally intensive to run. In the experiments described here, Azure virtual machines were used train the models. Machine sizes and run times listed [here](reports/mogp_azure_runs.xlsx).

 The folder is intended for reference; code cannot be run unless clinical data is gathered by the user.

 **Download Data**: This analysis uses clinical scores from four clinical ALS cohorts, three of which are available to download publicly or upon request. See 2a below for versions used in manuscript.
 - AnswerALS (AALS): AALS is publicly available (download "Full Metadata" at data.answerals.org)
 - Clinical Trial of Ceftriaxone in ALS (CEFT): CEFT can be downloaded from National Institute of Neurological Disorders and Stroke (NINDS) (https://www.ninds.nih.gov/Current-Research/Research-Funded-NINDS/Clinical-Research/Archived-Clinical-Research-Datasets) by request
 - The Pooled Resource Open-Access ALS Clinical Trials (PRO-ACT): PRO-ACT can be downloaded by request (https://nctu.partners.org/ProACT)
 - Emory ALS Clinic database (EMORY): Restricted access at this time

## 1) Pre-processing
**Description**: Creates .csv matrices from .sas7bdat files - for friendlier use with Python scripts  
**Inputs**: Ceftriaxone data folder with SAS files, `data/raw_data/ceft`  
**Outputs**: Ceftriaxone data folder with CSVs, `data/processed_data/ceft`  
**Script**: [sas_to_csv.py](sas_to_csv.py)

## 2) Clean and process clinical data
### a. Process raw clinical data to matrix
**Description**: Harmonize clinical data to consistent format  
**Inputs**: Paths to each of the raw datafiles or folders  
- AnswerALS: `data/raw_data/aals` (version: Dec 22, 2020)
- Ceftriaxone: `data/processed_data/ceft`
- Emory: `data/raw_data/emory/emory_deidentified_04012020.xlsx` (version: Apr 1, 2020)
- PROACT: `data/raw_data/proact` (version:  Jan 4, 2016)

**Outputs**: Processed static and timeseries datafiles: `static_proact_death.csv`, `timeseries_all_alsfrsr.csv`, `timeseries_proact_fvcp.csv`      
**Script**: [clean_clinical_data.py](clean_clinical_data.py)

### b. Process data into matrices for MoGP experiments
**Description**: Generate numpy matrices for all experiments   
**Inputs**: Paths to each of the processed clinical datafiles: `static_proact_death.csv`, `timeseries_all_alsfrsr.csv`, `timeseries_proact_fvcp.csv`   
 **Outputs**: pickled data files for each model, in `data/model_data`  
 **Script**: [process_data_for_mogp_experiments.py](process_data_for_mogp_experiments.py)

## 3) Run MoGP Experiments
**Description**: Generate trained models for all experiments. MoGP can be computationally intensive. Runtimes for models found in [mogp_azure_runs.xlsx](reports/mogp_azure_runs.xlsx)     
**Inputs**: pickled data files for each model, in `data/model_data`  
**Outputs**: trained model files, in `data/model_data`   
**Script**: [run_mogp_experiments.py](run_mogp_experiments.py)

## 4) Figures - see Jupyter notebooks for more information

**Full MoGP Trajectories:** Fig 1 - [plot_mogp_full_panel_figure.ipynb](plot_mogp_full_panel_figure.ipynb)

**Study Summary Statistics:** Table 1 -
[summ_stats_mogp_table.ipynb](summ_stats_mogp_table.ipynb)

**Trajectory Linearity:** Fig 2 -
[plot_mogp_linearity.ipynb](plot_mogp_linearity.ipynb)

**Interpolation/Prediction:** Fig 3 - [sparsity_prediction_process.py](sparsity_prediction_process.py), [plot_mogp_pred_sparse.ipynb](plot_mogp_pred_sparse.ipynb)

**Reference Model:** Fig 4 -
[plot_reference_model_experiment.ipynb](plot_reference_model_experiment.ipynb)

**Kaplan-Meier Survival Curves:** Fig 5 -
[plot_survival_curves.ipynb](plot_survival_curves.ipynb)

**Alternate outcomes:** Fig 6 -
[plot_alternate_outcomes.ipynb](plot_alternate_outcomes.ipynb)
