#!/bin/bash

# https://docs.alliancecan.ca/wiki/Using_GPUs_with_Slurm

#SBATCH --account=xxxxxxxx
#####--- S-BATCH --nodes=1  # to get a whole node
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=32000M               # memory (per node)
#SBATCH --cpus-per-task=6         # CPU cores/threads
#SBATCH --time=0-03:00            # time (DD-HH:MM)
#SBATCH --mail-user=xxxxxxx
#SBATCH --mail-type=ALL
                         # you can use 'nvidia-smi' for a test

#Old: S-BATCH --gres=gpu:1              # Number of GPUs (per node)


#$ srun --jobid yyyyyyy --pty watch -n 30 nvidia-smi
#$ srun --jobid yyyyyyy nvidia-smi
# srun --jobid yyyyyyy --pty tmux new-session -d 'htop -u $USER' \; split-window -h 'watch nvidia-smi' \; attach

__conda_setup="$('/home/xxxxx/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/xxxxx/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/xxxxx/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/xxxxx/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

python --version
conda activate pytorch
python --version
python t5trainer.py


