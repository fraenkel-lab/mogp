# MoGP: Mixture of Gaussian Processes Model for Sparse Longitudinal Data

## Background
MoGP is a a flexible framework for clustering longitudinal data to identify trajectory patterns

![Visual Abstract](docs/mogp_visual_abstract.jpg)

The model leverages two Bayesian nonparametric methods:
- **Gaussian process regression**: Learns trajectories from data, enabling the model to capture a wide variety of progression patterns; Does not require the specification of a particular functional form
- **Dirichlet process clustering**: Determines a number of clusters that is consistent with the number of trajectory trends observed in the data; Does not require the specification of a number of clusters a priori

The model was developed in the context of ALS disease progression modeling where all clinical scores are expected to decline over  time, and therefore includes an option for using an inductive bias towards monotonic decline.

## Tutorials
Tutorials for model usage can be found here:
- [Tutorial: Training a MoGP Model](example/tutorial_train_mogp_model.ipynb)
- [Tutorial: Using the ALSFRS-R Reference Model](example/tutorial_reference_model_predictions.ipynb)

We also provide a pre-trained reference model for ALSFRS-R scores that can be downloaded here: http://fraenkel.mit.edu/mogp/

## Installation
MoGP currently requires $\geq$`Python 3.8`.

Install using `pip`:
```
pip install mogp
```

If this does not work, you can also install from within the MoGP repository:
```
git clone https://github.com/fraenkel-lab/mogp
cd mogp/
python setup.py install --user
```
## Relevant Citations
Full article can be found here:  
Divya Ramamoorthy, Kristen Severson, Soumya Ghosh, Karen Sachs, Answer ALS, Jonathan D. Glass, Christina N. Fournier, Pooled Resource Open-Access ALS Clinical Trials Consortium, ALS/MND Natural History Consortium, Todd M. Herrington, James D. Berry, Kenney Ng & Ernest Fraenkel. Identifying patterns in amyotrophic lateral sclerosis progression from sparse longitudinal data. *Nat Comput Sci* **2**, 605–616 (2022). https://doi.org/10.1038/s43588-022-00299-w

Summary article can be found here:  
Divya Ramamoorthy & Ernest Fraenkel. Machine learning approach finds nonlinear patterns of neurodegenerative disease progression. *Nat Comput Sci* **2**, 565–566 (2022). https://doi.org/10.1038/s43588-022-00300-6

## Changelog
2022-12-05: Released mogp==1.0.0 - updated dependencies for python package; now requires python >=3.8
