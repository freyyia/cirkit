#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N learncircuit              
#$ -cwd                  
#$ -l h_rt=00:24:00 
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

# load environment
source venv/bin/activate


# Run the program
python ./notebooks/simple-circuit.py
