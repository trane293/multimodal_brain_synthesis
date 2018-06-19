#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks=1              # number of MPI processes
#SBATCH --cpus-per-task=24      # 24 cores on cedar nodes
#SBATCH --gres=gpu:lgpu:4            # special request for GPUs
#SBATCH --account=rrg-hamarneh
#SBATCH --mem=0                 # give all memory you have in the node
#SBATCH --time=1-00:00         # time (DD-HH:MM)
#SBATCH --job-name=train_mms
#SBATCH --output=slurm.%N.%j.out
#SBATCH --mail-user=asa224@sfu.ca
#SBATCH --mail-type=ALL

# load the module
module load python/2.7.14
module load cuda/8.0.44

# make sure bachrc is loaded
source ~/.bashrc
echo $LD_LIBRARY_PATH

export KERAS_BACKEND=theano

# run the command
THEANO_FLAGS=device=cuda0 python main_file.py --dir ./RESULTS --exp 1 --c ./RESULTS/split0/model
sleep 1 # pause to be kind to the scheduler
