#!/bin/bash
#SBATCH --job-name=autoencoderopt    # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
#SBATCH --partition=mcs.default.q

module load python
pip install -r requirements.txt
export TASK=$(ls datasets | sed -n $SLURM_ARRAY_TASK_ID'p')
python opt.py
