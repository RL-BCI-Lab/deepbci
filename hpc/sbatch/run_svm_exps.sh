#!/bin/bash
#
#SBATCH --job-name="svm_bci_exps"
#SBATCH --partition=Orion
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2GB
#SBATCH --cpus-per-task=36
#
#    ===== Main =====

cd /users/bpoole16/deep-bci/scripts/classification
python run_exps.py \
 --cpus=36 \
 --cpus_per_task=1 \
 --gpus=0 \
 --gpus_per_task=0 \
 --exp-cfg configs/sklearn/svm/examples/cv-exps/t2t-exp.yaml \
 --def-cfg configs/sklearn/svm/examples/cv-exps/t2t-def.yaml\
 --method-type logocv
