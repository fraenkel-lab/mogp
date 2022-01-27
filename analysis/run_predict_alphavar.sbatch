#!/bin/bash
#SBATCH -n 20 #Request 8 tasks (cores)
#SBATCH -N 1 #Request 1 node
#SBATCH --mem=1GB #Request 4G of memory - total memory per node
#SBATCH -o slurm/output_%j.txt #redirect output to output_JOBID.txt
#SBATCH -e slurm/error_%j.txt #redirect errors to error_JOBID.txt

while getopts a:b:c:d:e:f:g: flag
do
    case "${flag}" in

        a) exp=${OPTARG};;
        b) proj=${OPTARG};;
        c) kernel=${OPTARG};;
        d) run_by_seed=${OPTARG};;
        e) seed=${OPTARG};;
        f) tasknum=${OPTARG};;
        g) alpha_scale=${OPTARG};;
    esac
done



python run_mogp_experiments.py --exp $exp --proj $proj --kernel $kernel --run_by_seed $run_by_seed --seed $seed --tasknum $tasknum --alpha_scale $alpha_scale