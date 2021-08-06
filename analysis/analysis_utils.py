from model_postprocessing import check_model_monotonicity, split_nonmonotonic_clusters

import joblib
import numpy as np
import pandas as pd
from scipy.stats import linregress

from scipy.optimize import curve_fit
# import statsmodels.formula.api as smf


###########################################################################
#######                   Slope/Error Functions                     #######
###########################################################################


def linreg_fxn(varx, vary):
    """Simple linear regression"""
    mask = ~np.isnan(varx) & ~np.isnan(vary)
    # try:
    slope, intercept, _, _, _ = linregress(varx[mask], vary[mask])
    return slope, intercept
    # except:
    #     return np.nan


def pred_linreg_fxn(x_train, y_train, x_pred):
    """Predict y values given training and test vectors"""
    slope_train, intercept_train = linreg_fxn(x_train, y_train)
    y_pred = x_pred*slope_train + intercept_train
    # return y_pred
    return y_pred, slope_train, intercept_train



def sigmoid_d50(x, D50, dx):
    """Calculates y_prerdiction using 2-parameter sigmoidal model (Based off D50 ALS model)"""
    y = 48 / (1 + np.exp((x - D50) / dx))
    return y


def pred_sigmoid_fxn(x_train, y_train, x_pred):
    """Use sigmoid function to predict y values given training and test vectors"""
    d50_init = 5
    dx_init = 0.5
    p0 = [d50_init, dx_init]  # Initial guess, based on max/min values
    popt, pcov = curve_fit(sigmoid_d50, x_train, y_train, p0, method='dogbox', bounds=((0.1, 0.1), (75, 5)))
    y_pred = sigmoid_d50(x_pred, *popt)
    # return y_pred
    return y_pred, popt, pcov


def calc_y_pred_model(mod, x_real, cur_clust):
    """Calculate y_pred using given x and y data; return both original y value and predicted value"""
    pred_model = mod.obsmodel[cur_clust].model.predict(x_real.reshape(-1, 1))
    y_pred = pred_model[0].transpose()  # mean of prediction
    return y_pred


def rmse(predictions, targets):
    """Simple rmse calculation"""
    return np.sqrt(((predictions - targets) ** 2).mean())


def calc_error(y_real, y_pred):
    """Calculate error (rmse) between y_real and y_pred """
    if len(y_real) > 0:
        curr_err = rmse(y_pred, y_real)
    else:
        curr_err = np.nan
    return curr_err


def calc_y_pred(model_type, x_train, y_train=None, x_pred=None, mod=None, i=None):
    """Calculates y__predictions for  all baselines"""
    assert model_type in ['slope', 'sigmoid', 'rbf', 'linear'], 'model type {} not implemented'.format(
        model_type)  # all currently implemented baselines

    if model_type is 'slope':
        y_pred, _, _ = pred_linreg_fxn(x_train, y_train, x_pred)  # test x data identical to train x data for this exp
    elif model_type is 'sigmoid':
        y_pred, _, _ = pred_sigmoid_fxn(x_train, y_train, x_pred)
    # elif model_type is 'gp':
    #     y_pred = pred_single_gp(x_train, y_train, x_pred)
    # elif model_type is 'lme':
    #     assert mod is not None
    #     y_pred, _ = pred_lme_model(i, mod, x_pred)
    elif (model_type is 'rbf') or (model_type is 'linear'):
        assert mod is not None
        cur_clust = mod.z[i]
        y_pred = calc_y_pred_model(mod, x_pred, cur_clust)

    y_pred = y_pred.reshape(-1)
    return y_pred


# def train_lme_model(data):
#     """Train population-wide linear mixed effects model - with random slope and random intercept"""
#     X_full = data['XA']
#     Y_full = data['YA']

#     yflat = Y_full.reshape(-1)
#     xflat = X_full.reshape(-1)
#     pat_ind_o = np.arange(X_full.shape[0])
#     pat_ind = np.repeat(pat_ind_o, X_full.shape[1]).reshape(-1)

#     assert yflat.shape == xflat.shape
#     assert Y_full.shape == X_full.shape
#     assert pat_ind.shape == yflat.shape

#     df_data_long = pd.DataFrame({'id': pat_ind, 'x': xflat, 'Y': yflat})
#     df_data_long.dropna(how='any', inplace=True)

#     # linear mixed effects model (statsmodels)
#     md = smf.mixedlm("Y ~ x", df_data_long, groups=df_data_long["id"], re_formula="~x")
#     mdf = md.fit(method=["lbfgs"])

#     return mdf


# def pred_lme_model(i, mdf, x_pred):
#     """Predict LME model, given the index of the patient"""
#     group_int = mdf.params['Intercept']
#     group_slope = mdf.params['x']
#     y_pred_group = (group_int) + x_pred * (group_slope)
#     y_pred_indiv = (group_int + mdf.random_effects[i]['Group']) + x_pred * (mdf.random_effects[i]['x'] + group_slope)
#     return y_pred_indiv, y_pred_group


###########################################################################
#######                MoGP Analysis Functions                      #######
###########################################################################
def calc_slope_mogp_data(data):
    """Calculate average slope of each patient from data dictionary (exclude onset anchor)"""
    XA = data['XA'][:, 1:]  # Exclude anchor oonset
    # YA_nonorm = (data['YA'][:, 1:] * data['Y_std']) + data['Y_mean']  # Scale data to original
    YA = data['YA'][:, 1:]

    df_slope = pd.DataFrame(columns=['SI', 'slope'])
    for i, si in enumerate(data['SI']):
        slope_i, intercept_i = linreg_fxn(XA[i], YA[i])
        slope_i_ppm = slope_i / 12  # Calculate in points per month (x values currently in years)
        df_slope = df_slope.append({'SI': si, 'slope': slope_i_ppm}, ignore_index=True)
    df_slope.set_index('SI', inplace=True)

    return df_slope


def calc_clust_slope(model, data):
    """Calculate mean slope per MoGP cluster"""
    df_slope_si = calc_slope_mogp_data(data)
    df_slope_clust = pd.DataFrame(zip(data['SI'], model.z), columns=['SI', 'cluster']).set_index('SI')
    df_slope_clust = df_slope_clust.join(df_slope_si, how='left')
    df_slope_clust = pd.DataFrame(df_slope_clust.groupby('cluster')['slope'].mean())
    return df_slope_clust


def get_map_model(mod_path, mod_suffix, num_seeds=5, thresh=10, num_obs=20):
    """Select best MAP model from 5 seeds"""
    best_model_seed = None
    best_model = None
    best_ll = -1e12
    for seed in range(num_seeds):
        try:
            model = joblib.load(mod_path / '{}_seed_{}_MAP.pkl'.format(mod_suffix, seed))
            monot = check_model_monotonicity(model=model, thresh=thresh, num_obs=num_obs)
            if monot is True:
                if model.best_ll > best_ll:
                    best_ll = model.best_ll
                    best_model_seed = seed
                    best_model = model
            else:
                print('seed did not pass monotonicity test: {}'.format(seed))
        except FileNotFoundError:
            print('Seed not found: {}'.format(mod_path / '{}_seed_{}_MAP.pkl'.format(mod_suffix, seed)))
    if best_model == None:
        print('No models passed monotonicity test - check threshold: {}'.format(thresh))
    else:
        while check_model_monotonicity(best_model, thresh=thresh, num_obs=None) is False:
            # print('splitting cluster')
            split_nonmonotonic_clusters(best_model, thresh=thresh)
    print('best seed: {}, ll {}'.format(best_model_seed, best_ll))
    return best_model

class ModelSum:
    """Handy class to store all information about each model for prediction/sparsity experiments"""
    def __init__(self, task, task_num, seed, mod_type, err, num_clust, best_ll, si):
        self.err = err
        self.si = si
        self.num_clust = num_clust
        self.task = task
        self.mod_type = mod_type
        self.task_num = task_num
        self.seed = seed
        self.best_ll = best_ll

###########################################################################
#######                MoGP Plotting Functions                      #######
###########################################################################


def plot_mogp_by_clust(ax, model, data, k, data_flag=True, data_col='k', model_flag=True, model_col='b', model_alpha=0.2, gpy_pad=0.5):
    """Plot MoGP trajectory and data for individual cluster"""
    num_pat = 'n = {}'.format(model.allocmodel.Nk[k])
    if data_flag:
        assert (data is not None), 'missing data'
        XR = data['XA']
        YR = data['YA']
        ax.plot(XR[model.z == k].T[1:], YR[model.z == k].T[1:], 'o-', color=data_col, alpha=0.75)
    if model_flag:
        gpy_plt_xlim = model.obsmodel[k].X.max()+gpy_pad
        model.obsmodel[k].model.plot_mean(color=model_col, ax=ax, label=num_pat, plot_limits=[0, gpy_plt_xlim])
        model.obsmodel[k].model.plot_confidence(color=model_col, ax=ax, label='_nolegend_', alpha=model_alpha, plot_limits=[0, gpy_plt_xlim])
    return ax, num_pat


def plot_slope_by_clust(ax, model, k, lower_bound=0, upper_bound=1, estimate_x_val=3, slope_col='r'):
    """
    Calculate slope of MoGP trajectory between lower and upper bounds (default 0 to 1 years)
    Estimate difference between MoGP curve and calculated slope at the timepoint specified (estimate_x_val)
    """

    # Calculate slope
    x_slope = np.array([lower_bound, upper_bound])
    y_slope_pred_mean = model.obsmodel[k].model.predict(x_slope.reshape(-1, 1))[0]
    slope = ((y_slope_pred_mean[1] - y_slope_pred_mean[0]) / (x_slope[1] - x_slope[0]))[0]
    intercept = y_slope_pred_mean[0][0]

    x_slp_vals = np.array(ax.get_xlim())
    y_slp_vals = intercept + slope * x_slp_vals

    ax.plot(x_slp_vals, y_slp_vals, '--', color=slope_col, linewidth=3)


    # Estimate difference between slope prediction and MoGP at estimate_x_val years
    mogp_estim = model.obsmodel[k].model.predict(np.array([estimate_x_val]).reshape(-1, 1))[0][0][0]
    slope_estim = (intercept + slope * estimate_x_val)
    estim_diff = (mogp_estim - slope_estim)

    return estim_diff


def plot_largest_mogp_clusters(ax, model, data, disp_clust, color_palette, data_flag=True, model_flag=True, gpy_pad=0.5):
    """Plot x number of largest mogp clusters, where disp_clust indicates x"""
    nc = len(np.where(model.allocmodel.Nk > 0)[0])
    idx = np.argsort(-model.allocmodel.Nk)[0:nc]
    for i, k in enumerate(idx[:disp_clust]):
        ax, num_pat_k = plot_mogp_by_clust(ax, model, data, k, data_flag=data_flag, data_col=color_palette[i],
                                           model_flag=model_flag, model_col=color_palette[i], gpy_pad=gpy_pad)
    return ax


def format_mogp_axs(ax, max_x=8, x_step=2.0, y_label=[0,24,48], y_minmax=(-5, 53)):
    ax.set_xlim([0, max_x])
    ax.set_xticks(np.arange(0, max_x + 1, x_step))
    ax.set_yticks(y_label)
    ax.set_ylim(y_minmax)

    return ax


def format_panel_axs(ax, alph_lab, num_pat, k_alph_flag, fontsize_numpat=20, fontsize_alph=25):
    """Scale axes to original data, label with number of patients per cluster"""
    ax = format_mogp_axs(ax, max_x=8)

    ax.get_legend().remove()
    ax.text(0.02, 0.02, num_pat, transform=ax.transAxes, va='bottom', ha='left', fontsize=fontsize_numpat)
    if k_alph_flag:
        ax.text(0.97, 0.95, alph_lab, transform=ax.transAxes, va='top', ha='right', fontsize=fontsize_alph)
    return ax

