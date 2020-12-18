#!/bin/sh
#SBATCH --job-name=rter # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=23:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl1_48h-1G

python3 main.py --hidden_dim 100 --lr .001 --batch_size 32 --dropout .5
python3 main.py --hidden_dim 300 --lr .001 --batch_size 32 --dropout .5
