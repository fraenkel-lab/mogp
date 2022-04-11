#!/bin/bash
#SBATCH -n 16 #Request 8 tasks (cores)
#SBATCH -N 1 #Request 1 node
#SBATCH --mem=10GB #Request 4G of memory - total memory per node
#SBATCH -o slurm/output_%j.txt #redirect output to output_JOBID.txt
#SBATCH -e slurm/error_%j.txt #redirect errors to error_JOBID.txt
#SBATCH -p sched_mem1TB_centos7  #Run on sched_engaging_default partition
#SBATCH --exclude=node009,node011,node320
#SBATCH --time=24:00:00

exp='nonals'

while getopts p:s:k: flag
do
    case "${flag}" in

        p) proj=${OPTARG};;
        s) seed=${OPTARG};;
        k) kernel=${OPTARG};;
    esac
done


python run_mogp_experiments.py --exp $exp --proj $proj --kernel $kernel --seed $seed