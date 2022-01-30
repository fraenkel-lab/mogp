#!/bin/bash

while getopts b: flag
do
    case "${flag}" in
        b) proj=${OPTARG};;
    esac
done

exp='predict'
kernels_list=('rbf' 'linear')
# kernels_list=('rbf')
run_by_seed=True
seed_list=(0 1 2 3 4)
task_list=(0.25 0.50 1.0 1.5 2.0)
alpha_list=(0.1 0.5 2 10)

# exp='predict'
# kernels_list=('linear')
# run_by_seed=True
# seed_list=(0)
# task_list=(0.25)
# alpha_list=(0.1)

for kernel in "${kernels_list[@]}" ; do
	for seed in "${seed_list[@]}" ; do
		for task in "${task_list[@]}" ; do
			for alphasc in "${alpha_list[@]}" ; do
		    			sbatch run_predict_alphavar.sbatch -a $exp -b $proj -c $kernel -d $run_by_seed -e $seed -f $task -g $alphasc
			done
		done
	done
done
