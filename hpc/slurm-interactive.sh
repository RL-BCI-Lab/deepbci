#!/bin/bash
#SBATCH --partition=Hercules
#SBATCH --nodes=1
#SBATCH --ntasks=8 
#SBATCH --gres=gpu:TitanV:2
#SBATCH --time=3:00:00
#SBATCH --job-name=jupyter-notebook
#SBATCH --pty /bin/bash

# get tunneling info
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)
cluster=$(hostname -f | awk -F"." '{print $2}')

# print tunneling instructions jupyter-log
echo -e "

MacOS or linux terminal command to create your ssh tunnel
ssh -N -L ${port}:${node}:${port} ${user}@hpc.uncc.edu

Windows MobaXterm info
Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH login: $user
SSH port: 22

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"

# module load tensorflow/2.2-anaconda3-cuda10.2 openmpi/4.0.3
# conda activate base

# DON'T USE ADDRESS BELOW.
# DO USE TOKEN BELOW
echo "jupyter-notebook --no-browser --port=${port} --ip=${node}"

