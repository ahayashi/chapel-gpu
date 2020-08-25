#!/bin/bash
#SBATCH --job-name=bs
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

N=536870912

for i in 1 2 3 4 5;
do
    ./bs.baseline -nl 1 --n=$N --numTrials=10
    ./bs.gpu -nl 1 --n=$N --numTrials=10
    for ratio in 0 25 50 75 100;
    do
	./bs.hybrid -nl 1 --n=$N --numTrials=10 --CPUratio=$ratio
    done
done
