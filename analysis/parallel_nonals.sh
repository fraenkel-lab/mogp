#!/bin/bash

# proj_list=('adni' 'ppmi')
# seed_list=(0 1 2 3 4)
kernels_list=('rbf')
proj_list=('adni')
seed_list=(0)

for kernel in "${kernels_list[@]}" ; do
	for seed in "${seed_list[@]}" ; do
		for proj in "${proj_list[@]}" ; do
			sbatch run_nonals.sbatch -p $proj -s $seed -k $kernel
		done
	done
done
