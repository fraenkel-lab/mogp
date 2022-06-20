from model_postprocessing import check_model_monotonicity, split_nonmonotonic_clusters

import joblib
import numpy as np
import pandas as pd
from scipy.stats import linregress

from scipy.optimize import curve_fit
from scipy.stats import ks_2samp
import statsmodels.formula.api as smf
from mogp.neg_linear import *
from mogp import utils

import GPy
from sklearn.cluster import KMeans
from string import ascii_lowercase
import matplotlib
import matplotlib.pyplot as plt
import math
from scipy.stats import hypergeom



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
    """Calculates y_prediction using 2-parameter sigmoidal model (Based off D50 ALS model)"""
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


def quadratic(x, a, b, c):
    """Calculates y_prediction using 3-parameter quadratic model"""
    y = a*(x**2)+b*x+c
    return y

def pred_quad_fxn(x_train, y_train, x_pred):
    """Use quadratic function to predict y values given training and test vectors"""
    a_init = 1
    b_init = 1
    c_init = 1
    p0 = [a_init, b_init, c_init]  # Initial guess, based on max/min values
    popt, pcov = curve_fit(quadratic, x_train, y_train, p0, method='dogbox')
    y_pred = quadratic(x_pred, *popt)
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
    assert model_type in ['slope', 'sigmoid', 'rbf', 'linear', 'gp', 'lme', 'quad'], 'model type {} not implemented'.format(
        model_type)  # all currently implemented baselines

    if model_type is 'slope':
        y_pred, _, _ = pred_linreg_fxn(x_train, y_train, x_pred)  # test x data identical to train x data for this exp
    elif model_type is 'sigmoid':
        y_pred, _, _ = pred_sigmoid_fxn(x_train, y_train, x_pred)
    elif model_type is 'gp':
        y_pred = pred_single_gp(x_train, y_train, x_pred)
    elif model_type is 'quad':
        y_pred, _, _ = pred_quad_fxn(x_train, y_train, x_pred)
    elif model_type is 'lme':
        assert mod is not None
        y_pred, _ = pred_lme_model(i, mod, x_pred)
    elif (model_type is 'rbf') or (model_type is 'linear'):
        assert mod is not None
        cur_clust = mod.z[i]
        y_pred = calc_y_pred_model(mod, x_pred, cur_clust)

    y_pred = y_pred.reshape(-1)
    return y_pred


def train_lme_model(data):
    """Train population-wide linear mixed effects model - with random slope and random intercept"""
    X_full = data['XA']
    Y_full = data['YA']

    yflat = Y_full.reshape(-1)
    xflat = X_full.reshape(-1)
    pat_ind_o = np.arange(X_full.shape[0])
    pat_ind = np.repeat(pat_ind_o, X_full.shape[1]).reshape(-1)

    assert yflat.shape == xflat.shape
    assert Y_full.shape == X_full.shape
    assert pat_ind.shape == yflat.shape

    df_data_long = pd.DataFrame({'id': pat_ind, 'x': xflat, 'Y': yflat})
    df_data_long.dropna(how='any', inplace=True)

    # linear mixed effects model (statsmodels)
    md = smf.mixedlm("Y ~ x", df_data_long, groups=df_data_long["id"], re_formula="~x")
    mdf = md.fit(method=["lbfgs"])

    return mdf


def pred_lme_model(i, mdf, x_pred):
    """Predict LME model, given the index of the patient"""
    group_int = mdf.params['Intercept']
    group_slope = mdf.params['x']
    y_pred_group = (group_int) + x_pred * (group_slope)
    y_pred_indiv = (group_int + mdf.random_effects[i]['Group']) + x_pred * (mdf.random_effects[i]['x'] + group_slope)
    return y_pred_indiv, y_pred_group

def pred_single_gp(x_train, y_train, x_pred):
    """Predict single-patient gp, using MoGP model priorss"""
    y_mean = np.mean(y_train[~np.isnan(y_train)])
    y_std = np.std(y_train[~np.isnan(y_train)])
    if y_std == 0:
        # print('ystd 0', y_train)
        y_std = 1
        y_train = y_train - y_mean
    # assert (y_std > 0), 'no std; zscore error'

    y_train = (y_train - y_mean) / y_std

    mf = NegLinear(1, 1)
    kernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=4)

    kernel.lengthscale.set_prior(GPy.priors.Gamma.from_EV(4., 9.), warning=False)
    kernel.variance.set_prior(GPy.priors.Gamma.from_EV(1., .5), warning=False)

    #     m = GPy.models.GPRegression(x_train.reshape(-1,1), y_train.reshape(-1,1), kernel)
    m = GPy.models.GPRegression(x_train.reshape(-1, 1), y_train.reshape(-1, 1), kernel=kernel, mean_function=mf)

    m.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(0.75, 0.25 ** 2), warning=False)
    m.mean_function.set_prior(GPy.priors.Gamma.from_EV(2 / 3, 0.2), warning=False)

    m.optimize()
    y_pred = m.predict(x_pred.reshape(-1, 1))[0]

    y_pred = (y_pred * y_std) + y_mean

    return y_pred

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


def plot_mogp_by_clust(ax, model, data, k, data_flag=True, data_col='k', model_flag=True, model_col='b', model_alpha=0.2, gpy_pad=0.5, anchor=True):
    """Plot MoGP trajectory and data for individual cluster"""
    num_pat = 'n = {}'.format(model.allocmodel.Nk[k])
    if data_flag:
        assert (data is not None), 'missing data'
        XR = data['XA']
        YR = data['YA']
        if anchor:
            ax.plot(XR[model.z == k].T[1:], YR[model.z == k].T[1:], 'o-', color=data_col, alpha=0.75)
        else:
            ax.plot(XR[model.z == k].T[0:], YR[model.z == k].T[0:], 'o-', color=data_col, alpha=0.75)

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


def format_panel_axs(ax, alph_lab, num_pat, k_alph_flag, fontsize_numpat=20, fontsize_alph=25, max_x=8, x_step=2.0, y_label=[0,24,48], y_minmax=(-5, 53)):
    """Scale axes to original data, label with number of patients per cluster"""
    ax = format_mogp_axs(ax, max_x=max_x, x_step=x_step, y_label=y_label, y_minmax=y_minmax)

    ax.get_legend().remove()
    ax.text(0.02, 0.02, num_pat, transform=ax.transAxes, va='bottom', ha='left', fontsize=fontsize_numpat)
    if k_alph_flag:
        ax.text(0.97, 0.95, alph_lab, transform=ax.transAxes, va='top', ha='right', fontsize=fontsize_alph)
    return ax

def plot_mogp_panel(model, data, disp_clust=12, k_alph_flag=True, mogp_color='b', slope_color='r'):
    """Plot full panel, including calculating slope per cluster"""
    fig, axs = plt.subplots(math.ceil(disp_clust/4), 4, figsize=(20, 3*(math.ceil(disp_clust/4))), sharex=True, sharey=True)

    nc = len(np.where(model.allocmodel.Nk > 0)[0])
    idx = np.argsort(-model.allocmodel.Nk)[0:nc]
    
    df_disp_clust = pd.DataFrame(columns=['k', 'k_alph', 'estim_diff'])
    for i, k in enumerate(idx[:disp_clust]):
        if k_alph_flag:
            k_alph = ascii_lowercase[i]
        else:
            k_alph = None
        
        axs.flat[i], num_pat = plot_mogp_by_clust(axs.flat[i], model, data, k, data_col='k', model_col=mogp_color) 
        estim_diff = plot_slope_by_clust(axs.flat[i], model, k, slope_col=slope_color) 
        axs.flat[i] = format_panel_axs(axs.flat[i], k_alph, num_pat, k_alph_flag)
        
        df_disp_clust = df_disp_clust.append({'k': k, 'k_alph': k_alph, 'estim_diff': estim_diff}, ignore_index=True)
    
    return fig, axs, df_disp_clust

def get_clust_num_perc(model, vis_perc=0.9):
	"""for plotting extended figures; get x numb clusters for % of patients"""
	nc = len(np.where(model.allocmodel.Nk > 0)[0])
	idx = np.argsort(-model.allocmodel.Nk)[0:nc]

	tot = model.allocmodel.Nk[idx].sum()
	cursum = 0
	i = 0
	while cursum < tot*vis_perc:
	    cursum += model.allocmodel.Nk[idx][i]
	    i+=1

	return i

def summ_cluster_params(model, data):
    # Rescale mean slope to original scale (otherwise, is z-score normalized)
    nc = len(np.where(model.allocmodel.Nk > 0)[0])
    idx = np.argsort(-model.allocmodel.Nk)[0:nc]

    clust_params = pd.DataFrame(columns=['clust','neg_linmap.A', 'rbf.lengthscale', 'Gaussian_noise.variance'])
    csize = pd.DataFrame()
    for k in idx:
        curclust = model.obsmodel[k].model
        clust_params = clust_params.append({'clust':k, 'neg_linmap.A':curclust[0], 'rbf.lengthscale':curclust[2], 'Gaussian_noise.variance':curclust[3]}, ignore_index=True)
        csize = csize.append({'clust':k, 'clust_size':model.allocmodel.Nk[k]}, ignore_index=True)

    clust_params.set_index('clust', inplace=True)
    clust_params['neg_linmap.A'] = clust_params['neg_linmap.A']*data['Y_std'] # use rescaled slopes
    clust_params = clust_params.join(csize.set_index('clust'), how='left')
    
    return clust_params

def elbow_k(df):
    """Plot k-means elbow"""
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        _ = kmeanModel.fit(df)
        distortions.append(kmeanModel.inertia_)

    plt.figure(figsize=(16,8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

def plot_k_clust(model, cparam, num_clusters_set, clust_thresh_size, col_palette, max_x=8, x_step=2.0, y_label=[0,24,48], y_minmax=(-5, 53), xlabel='Time from Onset (Years)', ylabel='ALSFRS-R Total'):
    """Analyze structure witin cluster parameters"""
    fig = plt.figure(figsize=(20,7), constrained_layout=True)
    wrs = [2, 0.2]+[1]*(math.ceil(num_clusters_set/2))
    gs = fig.add_gridspec(2, len(wrs), width_ratios=wrs)
    f_ax1 = fig.add_subplot(gs[:, 0])

    axs = []
    for i in range(0,2):
        for j in range(2,math.ceil(num_clusters_set/2)+2):
            axs.append(fig.add_subplot(gs[i,j]))

    # Plot kcluster
    kmeans = KMeans(n_clusters=num_clusters_set, random_state=0).fit(cparam[['neg_linmap.A', 'rbf.lengthscale']])
    cparam['k_label']=kmeans.labels_
    cparam_freq = cparam.groupby('k_label')['clust_size'].sum()/cparam['clust_size'].sum()
    collist = [col_palette[x] for x in kmeans.labels_]
    f_ax1.scatter(cparam['neg_linmap.A'],cparam['rbf.lengthscale'], s=cparam['clust_size']*2, color=collist, alpha=0.9)
    _ = f_ax1.set_xlabel('Negative Mean Slope')
    _ = f_ax1.set_ylabel('Lengthscale')


    # plot clusters on subplots below
    nkclust = len(np.unique(kmeans.labels_))
#     klist = np.unique(kmeans.labels_)
    # sort by progression rate
    klist = cparam.groupby('k_label')['neg_linmap.A'].mean().sort_values(ascending=False).index
    kalph = [ascii_lowercase[a] for a, j in enumerate(klist)]
    cparam['kalph']=cparam['k_label'].map(dict(zip(klist,kalph)))
    # klist = [2,0,3,1]

    for j, kclust in enumerate(klist):
        allclust = cparam.index[kmeans.labels_==kclust]
        cax = axs[j]
        for i, k in enumerate(allclust):
            if model.allocmodel.Nk[int(k)]>=clust_thresh_size:
                _, num_pat_k = plot_mogp_by_clust(cax, model, None, int(k), data_flag=False, data_col=col_palette[kclust],
                                                       model_flag=True, model_col=col_palette[kclust])
                _ = format_mogp_axs(cax, max_x=max_x, x_step=x_step, y_label=y_label, y_minmax=y_minmax)
        _ = cax.text(0.9, 0.9, '{}) {:.2f}%'.format(ascii_lowercase[j], cparam_freq.loc[kclust]*100), va='top', ha='right', transform = cax.transAxes)
        _ = cax.get_legend().remove()
        _ = cax.set_xlabel(xlabel)
        _ = cax.set_ylabel(ylabel)
    return cparam, cparam_freq, fig, f_ax1, axs

def cmerge(df_clust, clindf, feat, ptid):
    """Merge clust w clin"""
    clust_oshape = df_clust.shape[0]
    clinlist=[ptid,feat]
    df_clust = df_clust.merge(clindf[clinlist].drop_duplicates(), on=ptid, how='left')
    df_clust.dropna(subset=clinlist, inplace=True)
    assert df_clust.shape[0]<=clust_oshape, 'check duplicates, {}, {}'.format(df_clust.shape[0],clust_oshape)
    return df_clust

def check_clust_freq(df_clust, feat, ptid, clustid='cluster'):
    """Calc clust frequency, including hypergeom pvals"""
    clust_size = df_clust.groupby(clustid)[feat].count()
    clust_size.name='clust_size'
    
    df_cfreq = pd.DataFrame(df_clust.groupby(clustid)[feat].value_counts())
    df_cfreq.columns=['counts']
    df_cfreq = df_cfreq.reset_index().pivot(index=clustid, columns=feat, values='counts')

    df_cfreq = df_cfreq.join(clust_size)
    df_cfreq.sort_values(by='clust_size', ascending=False, inplace=True)
    
    # add frequencies
    for cf in df_clust[feat].dropna().unique():
        df_cfreq['{}_freq'.format(cf)]=df_cfreq[cf]/clust_size
        
    for kclust in df_cfreq.index:
        for cf in df_clust[feat].dropna().unique():
            drawn_succcess = df_cfreq.loc[kclust][cf]
            popsuccess = df_cfreq[cf].sum()
            popsize = df_cfreq['clust_size'].sum()
            sampsize = df_cfreq.loc[kclust]['clust_size']
            prb = hypergeom.pmf(drawn_succcess, popsize, popsuccess, sampsize)
            df_cfreq.loc[kclust, '{}_pval'.format(cf)]=prb
    df_cfreq.sort_values(by='clust_size', ascending=False, inplace=True)
    
    return df_cfreq

def gen_df_err_extend(cur_data, cur_mod, random_assignment=False, rand_seed=0):
    """Calculate error between patient real data and trajectory mean function of predicted cluster"""
    if random_assignment:
        np.random.seed(rand_seed)
        clust_list = (cur_mod.z).copy()
        np.random.permutation(clust_list)  
    
    df_err_extend = pd.DataFrame(columns=['full_id', 'id', 'proj', 'idx', 'clust', 'err'])

    for i in range(0,len(cur_data['SI'])):
        cur_x = cur_data['XA'][i]
        cur_x = cur_x[~np.isnan(cur_x)]
        cur_y = cur_data['YA'][i]
        cur_y = cur_y[~np.isnan(cur_y)]
        full_cur_id = cur_data['SI'][i]
        cur_id = cur_data['SI'][i].split('_')[0]
        cur_proj = cur_data['SI'][i].split('_')[1]

        if random_assignment:
            clust_assign = clust_list[i]
        else:
            rank_clust, _ = utils.rank_cluster_prediction(cur_mod, cur_x, cur_y)
            clust_assign = rank_clust[0]
            
        y_pred_mean = calc_y_pred_model(cur_mod, cur_x, clust_assign)
        err = calc_error(cur_y, y_pred_mean)
        df_err_extend = df_err_extend.append({'full_id':full_cur_id, 'id':cur_id, 'proj':cur_proj, 'idx':i, 'clust':clust_assign, 'err': err}, ignore_index=True)
    
    return df_err_extend

###########################################################################
#######                Linearity  Functions                         #######
###########################################################################

def rank_err(data_mod, mod_dict):
    """Calculate model error for all models in model dict"""
    # checks that order of patients is identical for both models, and that correct data dictionary is passed
    np.testing.assert_array_equal(mod_dict['linear'].Y, mod_dict['rbf'].Y, err_msg='lin and rbf models not trained on same data')
    np.testing.assert_array_equal(mod_dict['linear'].X, mod_dict['rbf'].X, err_msg='lin and rbf models not trained on same data')
    np.testing.assert_array_equal(mod_dict['rbf'].X, data_mod['XA'], err_msg='mogp models and data dictionary not trained on same data') 

    err_dict = {}
    for key in mod_dict.keys():
            err_dict[key] = calc_rmse_full_mod(data=data_mod, model_type=key, mod=mod_dict[key])

    return err_dict

def calc_rmse_full_mod(data, model_type=None, mod=None):
    """Calculate RMSE for model"""
    num_patients = len(data['SI'])
    err_arr = np.zeros((num_patients, 1))
    for i in range(0,num_patients):
        y_real, y_pred_mean = calc_y_pred_full_data(data=data, i=i, model_type=model_type, mod=mod)
        err_arr[i] = calc_error(y_real, y_pred_mean)
        
    return err_arr

def calc_y_pred_full_data(data, i, model_type=None, mod=None):
    """Pass in all X and Y model data"""
    x_real = data['XA'][i]
    x_real = x_real[~np.isnan(x_real)]
    y_real = data['YA'][i]
    y_real = y_real[~np.isnan(y_real)]
    
    y_pred = calc_y_pred(model_type=model_type, x_train=x_real,y_train=y_real, x_pred=x_real, mod=mod, i=i)
    assert y_real.shape == y_pred.shape, 'prediction and original y vectors not same shape'
        
    return y_real, y_pred
    
def calc_perc_nonlin_thresh(mod_err_comp, mod_err_rbf, err_thresh=0.1):
    """Calculate difference in error (rmse) between comparison model (comp model - rbf model)"""
    err_diff = mod_err_comp - mod_err_rbf
    perc_above = (sum(err_diff > err_thresh)/len(err_diff))[0]
    perc_mid = (sum((-err_thresh < err_diff) & (err_diff < err_thresh))/len(err_diff))[0]
    perc_below = (sum(-err_diff > err_thresh)/len(err_diff))[0]
    
    return err_diff, perc_above, perc_mid, perc_below


def gen_err_nonlin(map_dict, projects, data_path, experiment_name):
    """Generate dataframe of RMSE error between original data and model mean functions"""
    df_err = pd.DataFrame(columns=['project', 'kernel', 'err'])
    for proj in projects:
        mod_dict = map_dict[proj]
        proj_data = joblib.load(data_path / 'data_{}_{}.pkl'.format(proj, experiment_name))
        
        err_dict = rank_err(proj_data, mod_dict) 
        for key in err_dict.keys():
            df_key = pd.DataFrame(err_dict[key], columns=['err'])
            df_key['kernel']=key
            df_key['project']=proj
            df_err = df_err.append(df_key)
    
    return df_err

def gen_clust_summ_agg(proj, map_dict, proj_data, comp='slope'):
    """Rank clusters in order of linearity (MoGP vs comp1 Model)"""
    err_dict = rank_err(proj_data, map_dict)
    assert (comp in err_dict.keys()), 'missing comp in errdict: {}'.format(comp)
    err_diff = err_dict[comp]-err_dict['rbf']
    df_clust_sum = pd.DataFrame(zip(map_dict['rbf'].z, err_diff[:,0], proj_data['SI']), columns=['rbf_clust_num', 'err_diff', 'SI'])
    df_cluster_aggregate = df_clust_sum.groupby('rbf_clust_num')[['err_diff']].mean().join(df_clust_sum['rbf_clust_num'].value_counts())
    df_cluster_aggregate.sort_values('err_diff', ascending=False, inplace=True)
    
    return df_cluster_aggregate

def format_perc_table(df_perc_diff, label_projs):
    """Format table showing percent differences between models at varying RMSE thresholds"""
    df_perc = df_perc_diff.pivot(index='project', columns='thresh').round(2)['perc_above']
    df_perc.rename(index=label_projs, inplace=True)
    df_perc = df_perc.applymap(lambda x : '{:.2f}%'.format(x))
    return df_perc

