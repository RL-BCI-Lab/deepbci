#!/bin/sh
#
# ======= PBS OPTIONS ======= (user input required)
#
### Specify queue to run
#PBS -q titan
### Set the job name
#PBS -N dbci-exps
### Specify the # of cpus for your job.
#PBS -l nodes=1:ppn=2:gpus=1,mem=16gb
#PBS -l walltime=199:99:99
### pass the full environment
#PBS -V
#
# ===== END PBS OPTIONS =====

### run job
# module load openmpi/3.1.2
# module load tensorflow/2.0-anaconda3-cuda10.0

conda activate tf

cd $PBS_O_WORKDIR
cd ../experiments/classification && python run_exps.py --exp-cfg $1 --def-cfg $2

conda deactivate
