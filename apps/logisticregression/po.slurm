#!/bin/bash
#SBATCH --job-name=lr
#SBATCH --partition=commons
#SBATCH -N 1
#SBATCH --cpus-per-task=24
#SBATCH --threads-per-core=1
#SBATCH --time=03:00:00
#SBATCH --gres=gpu
#SBATCH --mail-user=ahayashi@rice.edu
#SBATCH --mail-type=ALL


cd $SLURM_SUBMIT_DIR
export CHPL_LAUNCHER=slurm-gasnetrun_ibv
export GASNET_PHYSMEM_MAX=1G
export CHPL_LAUNCHER_WALLTIME=03:00:00

./lr.baseline -nl 1 --nFeatures=65536 --nSamples=32 --nIters=1 --numTrials=10
./lr.gpu -nl 1 --nFeatures=65536 --nSamples=32 --nIters=1 --numTrials=10
for ratio in 0 25 50 75 100;
do
    ./lr.hybrid -nl 1 --nFeatures=65536 --nSamples=32 --nIters=1 --numTrials=10 --CPUratio1=$ratio --CPUratio2=$ratio
done
