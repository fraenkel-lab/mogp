from setuptools import setup

setup(
    name='mogp',
    version='1.0',
    description='Mixture of Gaussian Processes Model for Sparse Longitudinal Data',
    url='https://github.com/fraenkel-lab/mogp',
    python_requires='>=3.6',
    packages=['mogp'],
    # package_data={'mogp': ['data/reference_model.pkl']},

    install_requires=[
        'GPy>=1.9.8',
        'scipy>=1.3.0',
        'numpy>=1.16.4',
        'scikit-learn>=0.21.2',
        'sklearn>=0.0',
        'matplotlib>=3.1.1'],

    author='Divya Ramamoorthy',
    author_email='divyar@mit.edu',
    license='MIT'
)