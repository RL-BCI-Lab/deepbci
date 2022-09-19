#!/bin/bash
#
#SBATCH --job-name="svm_bci_exps"
#SBATCH --partition=Orion
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=3GB
#SBATCH --cpus-per-task=36
#
#    ===== Main =====

cd /users/bpoole16/deepbci/scripts/classification
python run_exps.py \
 --cpus=32 \
 --cpus_per_task=1 \
 --gpus=0 \
 --gpus_per_task=0 \
 --exp-cfg exps/v2-ErrP-variations-March-3-2022/logocv-v2-svm-async/configs/exp.yaml \
 --def-cfg exps/v2-ErrP-variations-March-3-2022/logocv-v2-svm-async/configs/exp-def.yaml \
 --method-type logocv
