import numpy as np
import matplotlib

###########################################################################
#######         Functions to Interact with MoGP Model               #######
###########################################################################


def rank_cluster_prediction(model, x_new, y_new):
    """Rank clusters with new x/y data by log likelihood

    Arguments:
        model (mogp.MoGP_constrained): model to use for cluster prediction
        x_new (np.array): explanatory variable (time since symptom onset) for new patient (length: # time points)
        y_new (np.aray): responses (clinical scores) for new patient (length: # time points)

    Returns:
        rank_cluster (np.array): sorted array with most likely cluster prediction first
        cluster_ll (np.array): corresponding log likelihoods for clusters in rank_cluster
    """
    p = model.predict(x_new, y_new)
    act_clust_ids = np.where(model.allocmodel.Nk > 0)[0]
    p_act_clust = p[act_clust_ids]

    rank_idx = np.argsort(p_act_clust)[::-1]
    rank_cluster = act_clust_ids[rank_idx]
    cluster_ll = p_act_clust[rank_idx]

    return rank_cluster, cluster_ll


def generate_toy_data(seed=0):
    """Generate toy data with 3 clusters

    Arguments:
        seed (int): random seed to initialize data generation for x_value
    """
    np.random.seed(seed=seed)
    X_toy = np.sort(np.vstack([np.random.uniform(0.1, 7., (5, 4)), np.random.uniform(3., 6., (5, 4)),
                               np.random.uniform(0.1, 3., (5, 4)), np.random.uniform(0.1, 4., (5, 4))]))
    noise = np.random.randn(X_toy.shape[0], 1) * 4
    Y_slow = 48 / (1 + np.exp(-2 * (-X_toy + 6))) + noise
    Y_med = 48 / (1 + np.exp(-2 * (-X_toy + 3))) + noise
    Y_fast = 48 / (1 + np.exp(-2 * (-X_toy + 1))) + noise
    X = np.vstack([X_toy, X_toy, X_toy])
    Y = np.vstack([Y_slow, Y_med, Y_fast])
    return X, Y


def plot_mogp(ax, model, display_all_clusters=True, num_clust_disp=None):
    """Light wrapper for GPy model plotting functions to plot mixture model - will plot the largest clusters

    Arguments:
        ax (matplotlib.axes): axes for plot
        model (mogp.MoGP_constrained): model to visualize
        display_all_clusters (bool): flag indicating if all clusters should be displayed
        num_clust_disp (int): if do not want all clusters, provide the number of clusters to display

    Returns:
        ax (matplotlib.axes): axes for plot
    """
    nc = len(np.where(model.allocmodel.Nk > 0)[0])
    idx = np.argsort(-model.allocmodel.Nk)[0:nc]

    if display_all_clusters is False:
        assert num_clust_disp is not None, 'provide number of clusters to display'
        disp_clust = num_clust_disp
    else:
        disp_clust = nc

    cmap = matplotlib.cm.get_cmap('rainbow')
    colors = [cmap(i/disp_clust) for i in range(disp_clust)]

    for i, k in enumerate(idx[:disp_clust]):
        k = idx[i]
        num_pat = 'n = {}'.format(model.allocmodel.Nk[k])
        _ = model.obsmodel[k].model.plot_mean(ax=ax, label=num_pat, color=colors[i])
        _ = model.obsmodel[k].model.plot_confidence(ax=ax, label='_nolegend_', color=colors[i])
    return ax
