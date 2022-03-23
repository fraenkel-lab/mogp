#!/bin/bash
#SBATCH -n 16 #Request 8 tasks (cores)
#SBATCH -N 1 #Request 1 node
#SBATCH --mem=10GB #Request 4G of memory - total memory per node
#SBATCH -o slurm/output_%j.txt #redirect output to output_JOBID.txt
#SBATCH -e slurm/error_%j.txt #redirect errors to error_JOBID.txt
#SBATCH -p sched_mem1TB_centos7  #Run on sched_engaging_default partition
#SBATCH --exclude=node009,node011
#SBATCH --time=02:00:00

while getopts a:b:c:d:e:f:g:h:i: flag
do
    case "${flag}" in

        a) exp=${OPTARG};;
        b) proj=${OPTARG};;
        c) kernel=${OPTARG};;
        d) run_by_seed=${OPTARG};;
        e) seed=${OPTARG};;
        f) tasknum=${OPTARG};;
        g) alpha_scale=${OPTARG};;
		h) minnum=${OPTARG};;
        i) expname=${OPTARG};;
    esac
done


python run_mogp_experiments.py --exp $exp --proj $proj --kernel $kernel --run_by_seed $run_by_seed --seed $seed --tasknum $tasknum --alpha_scale $alpha_scale --minnum $minnum --expname $expname