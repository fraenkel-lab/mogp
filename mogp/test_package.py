import pytest
import mogp
import shutil
import numpy as np
import joblib
import matplotlib.pyplot as plt


@pytest.fixture(scope='session')
def save_dir(tmp_path_factory):
    my_tmpdir = tmp_path_factory.mktemp("./test_temp")
    yield my_tmpdir 
    shutil.rmtree(str(my_tmpdir))

def test_tutorial_mogp(save_dir):
    X, Y = mogp.utils.generate_toy_data(seed=0)

    # Add onset anchor to matrix
    num_pat = X.shape[0]
    onset_anchor = 48 # normal onset anchor for ALSFRS-R scores
    X = np.hstack((np.zeros((num_pat,1)), X))
    Y = np.hstack((onset_anchor * np.ones((num_pat,1)), Y))
    # Provide output directory for model

    # Train model
    print(save_dir)
    mix = mogp.MoGP_constrained(X=X, Y=Y, alpha=1., num_iter=5, savepath=save_dir, rand_seed=0, normalize=True)
    mix.sample()

def test_nan_mogp(save_dir):
    # Make sample toy data - only using 3 patients (two with nans, one without)
    X_test = [1,2,np.nan,4]
    Y_test = [50, 40, np.nan, 20]

    X2_test = [1,2,4, np.nan]
    Y2_test = [50, 40, 20, np.nan]

    X3_test = [1,2,3,4]
    Y3_test = [50, 40, 30, 20]

    X = np.vstack([X_test, X2_test, X3_test])
    Y = np.vstack([Y_test, Y2_test, Y3_test])

    # Train MoGP model. For this test, num_init_clusters was set to 1 because we only made 3 simulated patients. You should delete this parameter when actually running model (for a dirichlet process, one must assign initial (largely random) cluster membership, and then the model iteratively refines actual cluster membership.  
    mix = mogp.MoGP_constrained(X=X, Y=Y, alpha=1., num_iter=5, savepath=save_dir, rand_seed=0, normalize=True, num_init_clusters=1)
    mix.sample()

def test_refmodel():
    reference_model = joblib.load('../example/mogp_reference_model.pkl')
    Xi_new = np.array([0.25, 0.5, 1, 2.4])
    Yi_new = np.array([46, 44, 41, 19])
    cluster_list, cluster_ll = mogp.utils.rank_cluster_prediction(reference_model, Xi_new, Yi_new)
    cur_clust = cluster_list[0] # Select most likely cluster (first cluster in ranked list)
    fig, ax = plt.subplots(figsize=(8,5))

    # Plot GP model for selected cluster
    _ = reference_model.obsmodel[cur_clust].model.plot_confidence(ax=ax, label='GP Confidence')
    _ = reference_model.obsmodel[cur_clust].model.plot_mean(ax=ax, label='GP Mean')

    # Plot input new data
    _ = ax.plot(Xi_new, Yi_new, 'o', color='g', label='Input Data')

    # Format plot
    _ = ax.set_xlim(0)
    _ = ax.legend()

# def test_webdownload():
