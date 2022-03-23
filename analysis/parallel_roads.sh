#!/bin/bash

while getopts p: flag
do
    case "${flag}" in
        p) proj=${OPTARG};;
    esac
done

seed_list=(0)
exp_list=('roadsnorm_noanchor')
exp='roads'
kernels_list=('rbf')

# kernels_list=('rbf' 'linear')
# seed_list=(0 1 2 3 4)
# exp_list=('roadsnorm_noanchor' 'roadsnorm_achor' 'alsfrst_noanchor' 'alsfrst_anchor')

# # Don't need below; using to fit pre-existing sbatch
run_by_seed=True
task=1.0
alphasc=1.0
minnum='min3'


for expname in "${exp_list[@]}" ; do
	for kernel in "${kernels_list[@]}" ; do
		for seed in "${seed_list[@]}" ; do
		    sbatch run_predict_alphavar.sbatch -a $exp -b $proj -c $kernel -d $run_by_seed -e $seed -f $task -g $alphasc -h $minnum -i $expname
		    # sh run_predict_alphavar.sbatch -a $exp -b $proj -c $kernel -e $seed -i $expname
		done
	done
done
