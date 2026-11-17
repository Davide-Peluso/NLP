#!/bin/bash
# These are the parameters for the job!

#SBATCH --job-name=sentiment_analysis_twitch                                  
#SBATCH --partition=dsba                                  
#SBATCH --cpus-per-task=96
#SBATCH --mem=49G                                 
#SBATCH --time=20:00:00                                      
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --gres=gpu:3                                            

module load modules/miniconda3
eval "$(conda shell.bash hook)"
conda activate

python sentiment_analysis_twitch.py                        