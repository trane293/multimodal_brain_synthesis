#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks=1              # number of MPI processes
#SBATCH --cpus-per-task=4      # 24 cores on cedar nodes
#SBATCH --gres=gpu:1            # special request for GPUs
#SBATCH --account=rrg-hamarneh
#SBATCH --mem=0                 # give all memory you have in the node
#SBATCH --time=150:00:00         # time (DD-HH:MM)
#SBATCH --job-name=exp1-train_multimodal-synthesizer
#SBATCH --output=slurm.%N.%j.out
#SBATCH --mail-user=asa224@sfu.ca
#SBATCH --mail-type=ALL

# load the module
module load python/2.7.14
module load cuda/8.0.44

which cuda
# make sure bachrc is loaded
source ~/.bashrc
echo $LD_LIBRARY_PATH

# run the command
export KERAS_BACKEND=theano
THEANO_FLAGS=device=cuda0 python main_file.py --dir ./RESULTS_ALL --exp 1
sleep 1 # pause to be kind to the scheduler
