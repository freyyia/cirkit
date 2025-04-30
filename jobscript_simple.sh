#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N learncircuit              
#$ -cwd                  
#$ -l h_rt=00:24:00 
# Request one GPU in the gpu queue:
#$ -q gpu 
#$ -l gpu=1
# Request x GB system RAM 
# the total system RAM available to the job is the value specified here multiplied by 
# the number of requested GPUs (above)
#$ -l h_vmem=10G
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit of 5 minutes: -l h_rt
#  memory limit of 1 Gbyte: -l h_vmem

# Initialise the environment modules
. /etc/profile.d/modules.sh


# Load Python
module load python
module load cuda

# load environment
source venv/bin/activate


# Run the program
python ./notebooks/simple_circuit.py
