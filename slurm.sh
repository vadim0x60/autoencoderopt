#!/bin/bash
#SBATCH --job-name=autoencoderopt    # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --time=48:00:00               # Time limit hrs:min:sec

module load python
pip install -r requirements.txt
export TASKS=$(ls datasets)
export TASK = TASKS{$SLURM_ARRAY_TASK_ID}
python opt.py | tee opt.out