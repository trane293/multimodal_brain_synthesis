#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks=1              # number of MPI processes
#SBATCH --cpus-per-task=24      # 24 cores on cedar nodes
#SBATCH --gres=gpu:lgpu:4            # special request for GPUs
#SBATCH --account=rrg-hamarneh
#SBATCH --mem=0                 # give all memory you have in the node
#SBATCH --time=02:00:00         # time (DD-HH:MM)
#SBATCH --job-name=cont_tms
#SBATCH --output=slurm.%N.%j.continuous.out
#SBATCH --mail-user=asa224@sfu.ca
#SBATCH --mail-type=ALL

# load the module
module load python/2.7.14
module load cuda/8.0.44

# make sure bachrc is loaded
source ~/.bashrc
echo $LD_LIBRARY_PATH

export KERAS_BACKEND=theano

EXP = 0
if [ -e ./RESULTS/split0/model ]
then
     # There is a checkpoint file, restart;
     THEANO_FLAGS=device=cuda0 python main_file.py --dir ./RESULTS --exp $EXP --c ./RESULTS/split0/model
else
     # There is no checkpoint file, start a new simulation.
     THEANO_FLAGS=device=cuda0 python main_file.py --dir ./RESULTS --exp $EXP
fi

# Resubmit if not all work has been done yet.
# if there is a model_100 file present
if [-e ./RESULTS/split0/training_complete.z ]
then
     # ---------------------------------------------------------------------
       echo "Job finished with exit code $? at: `date`"
     # ---------------------------------------------------------------------
else
     sbatch ${BASH_SOURCE[0]}
fi

