#!/bin/bash -l
 
################# Part-1 Slurm directives ####################
## Working dir
#SBATCH -D /users/tk2017/code/cirkit
## Environment variables
#SBATCH --export=ALL
## Output and Error Files
#SBATCH -o job-%j.output
#SBATCH -e job-%j.error
## Job name
#SBATCH -J gpu-test
## Run time: "hours:minutes:seconds", "days-hours"
#SBATCH --time=50:00:00
## Memory limit (in megabytes). Total --mem or amount per cpu --mem-per-cpu
#SBATCH --mem-per-cpu=20024
## GPU requirements
#SBATCH --gres gpu:1
## Specify partition
#SBATCH -p gpu

source "${flight_ROOT:-/opt/flight}"/etc/setup.sh
flight env activate conda@base
conda activate test_env



echo "$CONDA_PREFIX"


echo "Executing job commands, current working directory is $(pwd)"
echo "I was allocated the following GPU devices: $CUDA_VISIBLE_DEVICES" 

python notebooks/learning-circuit-with-pic-2.py
#python notebooks/simple_circuit.py