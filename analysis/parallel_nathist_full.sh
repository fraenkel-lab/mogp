#!/bin/bash

exp='full'
proj='nathist'
seed_list=(0 1 2 3 4)
#kernel='rbf'
kernel='linear'
run_by_seed=true

# Don't need below; using to fit pre-existing sbatch
task=1.0
alphasc=1.0
minnum='min3'
expname='nathistfull'

for seed in "${seed_list[@]}" ; do
    sbatch run_predict_alphavar.sbatch -a $exp -b $proj -c $kernel -d $run_by_seed -e $seed -f $task -g $alphasc -h $minnum -i $expname
done