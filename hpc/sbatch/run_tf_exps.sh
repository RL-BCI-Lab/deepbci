#!/bin/bash
#
#SBATCH --job-name="tf_bci_exps"
#SBATCH --partition=Hercules
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB
#SBATCH --gres=gpu:GTX1080ti:8
#
#    ===== Main =====

cd /users/bpoole16/deep-bci/scripts/classification
python run_exps.py \
 --cpus=8 \
 --cpus_per_task=1 \
 --gpus=8 \
 --gpus_per_task=1 \
 --exp-cfg configs/tensorflow/examples/cv-exps/t2t-exp.yaml \
 --def-cfg configs/tensorflow/examples/cv-exps/t2t-def.yaml \
 --method-type logocv
