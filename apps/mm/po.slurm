#!/bin/bash
#SBATCH --job-name=mm
#SBATCH --partition=commons
#SBATCH -N 1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=24
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu
#SBATCH --mail-user=ahayashi@rice.edu
#SBATCH --mail-type=ALL

cd $SLURM_SUBMIT_DIR
export CHPL_LAUNCHER=slurm-gasnetrun_ibv
export GASNET_PHYSMEM_MAX=1G
export CHPL_LAUNCHER_WALLTIME=02:00:00

./mm.baseline -nl 1 --n=1024 --numTrials=10
./mm.gpu -nl 1 --n=1024 --tiled=0 --numTrials=10

for ratio in 0 25 50 75 100;
do
    ./mm.hybrid -nl 1 --n=1024 --tiled=0 --numTrials=10 --CPUratio=$ratio
done

./mm.gpu -nl 1 --n=1024 --tiled=1 --numTrials=10

for ratio in 0 25 50 75 100;
do
    ./mm.hybrid -nl 1 --n=1024 --tiled=1 --numTrials=10 --CPUratio=$ratio
done
