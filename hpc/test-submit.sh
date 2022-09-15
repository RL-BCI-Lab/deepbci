#!/bin/bash
#
#BATCH --job-name="test"
#SBATCH --partition=Hercules
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=8GB
#
#    ===== Main =====

echo "This is a test!"
