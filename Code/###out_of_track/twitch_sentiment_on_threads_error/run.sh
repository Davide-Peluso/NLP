#!/bin/bash
# These are the parameters for the job!

#SBATCH --job-name=sent_t_v4                                       # Job name : change to your job name if you want
#SBATCH --partition=dsba                                       # Partition name : change   -> e.g. stud, dsba, etc.
#SBATCH --cpus-per-task=96
#SBATCH --mem=49G                                               # Memory : change
#SBATCH --time=20:00:00                                         # Time limit hrs:min:sec   
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --gres=gpu:3                                            # Request 2 GPUs   --> Change to max possible for your partition

module load modules/miniconda3
eval "$(conda shell.bash hook)"
conda activate

python set_t_v0.py                                # Change to your script name --> Will run the script in the current directory