import numpy as np
from sklearn.cluster import KMeans
from mogp.obsmodel import GPobs


def check_model_monotonicity(model, thresh=10, window=1, num_obs=20):
    """check if any jumps occur within largest clusters"""
    model_flag = True

    nc = len(np.where(model.allocmodel.Nk > 0)[0])
    idx = np.argsort(-model.allocmodel.Nk)[0:nc]
    if num_obs is not None:
        idx = idx[:num_obs]
    for i, k in enumerate(idx):
        obs = model.obsmodel[k]
        monot_flag = find_cluster_jumps(obs=obs, thresh=thresh, window=window)
        model_flag = model_flag and monot_flag

    return model_flag

def find_cluster_jumps(obs, thresh=10, window=1):
    """uses sliding windows to check if score jump greater than threshold occurs"""
    min_point = np.min(obs.X)
    while (min_point + window) < np.max(obs.X):
        y_pred_mean, _ = obs.model.predict(np.array([min_point, min_point + window]).reshape(-1, 1))
        if (y_pred_mean[1] - y_pred_mean[0]) > thresh:
            # obs.model.plot()
            return False
        min_point += 0.1

    else:
        return True

# def find_cluster_crossings(obs, step=0.1):
#     """uses gradient to check if inflection point occurs"""
#     max_x = obs.X.max()
#     xplt = np.arange(0, max_x, step).reshape(-1, 1)
#     y_pred_mean, _ = obs.model.predict(xplt)
#     y_pred_grad, _ = obs.model.predictive_gradients(xplt)
#     zero_crossings = np.where(np.diff(np.signbit(y_pred_grad), axis=0))[0]
#     if len(zero_crossings > 1):
#         return xplt[zero_crossings]
#     else:
#         return None


def split_nonmonotonic_clusters(model, rand_seed=0, thresh=10):
    """MoGP uses weak monotonic priors, which can lead to non-monotonic behavior in some sparse data; account for this by splitting non-monotonic functions"""
    nc = len(np.where(model.allocmodel.Nk > 0)[0])
    idx = np.argsort(-model.allocmodel.Nk)[0:nc]

    for i, k in enumerate(idx):
        obs = model.obsmodel[k]
        xmon = find_cluster_jumps(obs=obs, thresh=thresh)
        if xmon is False:
            # obs.model.plot()

            ind_z = np.where(model.z == k)[0]
            obs_x = model.X[ind_z]
            # obs_y = model.Y[ind_z]
            # plt.plot(obs_x[:, 1], obs_y[:, 1], 'o')

            # take non anchor-onset datapoint to cluster patients
            kmeans = KMeans(n_clusters=2, random_state=rand_seed).fit(obs_x[:, 1].reshape(-1,1))
            split_clust = np.array(kmeans.labels_)
            clust_1 = ind_z[np.where(split_clust == 0)]
            clust_2 = ind_z[np.where(split_clust == 1)]

            model.obsmodel[k] = None  # delete current cluster; add new clusters to end
            model.allocmodel.Nk[k] = 0
            generate_clust_obs(model, clust_1)
            generate_clust_obs(model, clust_2)

    # Normalize depending on model params
    model._apply_normalizer()


def generate_clust_obs(model, clust, optimize_params=True):
    """fit new GP models to split clusters"""
    xclust = model.X[clust]
    yclust = model.Y[clust]
    clust_Nk = xclust.shape[0]

    new_obs = GPobs(xclust[~np.isnan(xclust)].reshape(-1, 1),
                               yclust[~np.isnan(yclust)].reshape(-1, 1),
                               kernel=model.kernel, mean_func=model.mean_func, signal_variance=model.signal_variance,
                               signal_variance_fix=model.signal_variance_fix,
                               noise_variance=model.noise_variance, noise_variance_fix=model.noise_variance_fix)

    if optimize_params:
        new_obs.model.optimize()

    # Add new cluster to model
    model.allocmodel.Nk = np.hstack([model.allocmodel.Nk, clust_Nk])
    active_comp_ids = np.where(model.allocmodel.Nk > 0)[0]
    new_comp = active_comp_ids[-1]
    model.z[clust] = new_comp #update model.z. model.p not updated.
    model.obsmodel[new_comp] = new_obs # add new comp to obsmodel

    # new_obs.model.plot(label=new_comp)
    # score  np.log(model.allocmodel.Nk + 1e-16)
    # score[new_comp] += new_obs.ll #update score of new comp


