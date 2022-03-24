#!/bin/bash

# Usages:
# sh parallel_predict_alphavar.sh -e 'sparse' -b 'aals' -m 'min10'
# sh parallel_predict_alphavar.sh -e 'sparse' -b 'emory' -m 'min10'

# sh parallel_predict_alphavar.sh -e 'predict' -b 'aals' -m 'min4'
# sh parallel_predict_alphavar.sh -e 'predict' -b 'emory' -m 'min4'

# sh parallel_predict_alphavar.sh -e 'sparse' -b 'ceft' -m 'min4'
# sh parallel_predict_alphavar.sh -e 'sparse' -b 'aals' -m 'min4'
# sh parallel_predict_alphavar.sh -e 'sparse' -b 'emory' -m 'min4'
# sh parallel_predict_alphavar.sh -e 'sparse' -b 'proact' -m 'min4'

alpha_list=(1.0)
alphaset=False

while getopts e:b:m:a: flag
do
    case "${flag}" in
    	e) exp=${OPTARG};;
        b) proj=${OPTARG};;
		m) minnum=${OPTARG};;
		a) alphaset=${OPTARG};;
    esac
done

# exp='predict'
kernels_list=('rbf' 'linear')
# kernels_list=('rbf')
run_by_seed=True
seed_list=(0 1 2 3 4)
if $alphaset
then
	alpha_list=(0.1 0.5 2 10)
fi

if [[ "$exp" == "predict" ]]; then
    task_list=(2.0 1.5 1.0 0.5 0.25)
else
    task_list=(25 50 75)
fi

exp_name='alphavar'
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
		    			sbatch run_predict_alphavar.sbatch -a $exp -b $proj -c $kernel -d $run_by_seed -e $seed -f $task -g $alphasc -h $minnum -i $expname
			done
		done
	done
done
