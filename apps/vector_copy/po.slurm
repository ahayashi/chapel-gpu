#!/bin/bash
#SBATCH --job-name=vc
#SBATCH --partition=commons
#SBATCH -N 1
#SBATCH --time=00:15:00
#SBATCH --gres=gpu
#SBATCH --output=slurm.out
#SBATCH --error=slurm.error
#SBATCH --mail-user=ahayashi@rice.edu
#SBATCH --mail-type=ALL

cd $SLURM_SUBMIT_DIR
export CHPL_LAUNCHER=slurm-gasnetrun_ibv
export GASNET_PHYSMEM_MAX=1G
export CHPL_LAUNCHER_WALLTIME=00:15:00

./vc.baseline -nl 1 --n=536870912 --numTrials=10
./vc.gpu -nl 1 --n=536870912 --numTrials=10
for ratio in 0 25 50 75 100;
do
    ./vc.hybrid -nl 1 --n=536870912 --numTrials=10 --CPUratio=$ratio
done
