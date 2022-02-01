#!/bin/bash
#SBATCH --job-name=autoencoderopt    # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --time=96:00:00               # Time limit hrs:min:sec
#SBATCH --partition=mcs.default.q

export TASK=$(ls datasets | sed -n $SLURM_ARRAY_TASK_ID'p')
python opt.py
