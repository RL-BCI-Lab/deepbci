#!/bin/sh
#
# ======= PBS OPTIONS ======= (user input required)
#
### Specify queue to run
#PBS -q titan
### Set the job name
#PBS -N MC-3m-k6-fixed
### Specify the # of cpus for your job.
#PBS -l nodes=1:ppn=2:gpus=1,mem=15gb
#PBS -l walltime=199:99:99
### pass the full environment
#PBS -V
#
# ===== END PBS OPTIONS =====

### run job
module unload pymods perlmods
module load tensorflow/2.0-anaconda3-cuda10.0 openmpi/3.1.2
pip install --user pygame mpi4py
cd $PBS_O_WORKDIR
cd .. && python -m dqn.agents.oa.feedback -c train-config