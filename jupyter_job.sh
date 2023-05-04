#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=hipri 
#SBATCH --account=all
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00 # run for one day
#SBATCH --cpus-per-task=10
#SBATCH --output=/data/home/%u/jupyter-%j.log

bash ~/miniconda.sh -b -p ~/miniconda
eval "$(~/miniconda/bin/conda shell.bash hook)"
source ~/miniconda/bin/activate 
conda init

conda activate nlp
cat /etc/hosts
jupyter notebook --ip=0.0.0.0 --port=8888
