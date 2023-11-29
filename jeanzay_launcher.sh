#!/bin/bash
#SBATCH --job-name=ddpm # nom du job
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=16           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=20:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=ddpm%j.out # output file name
#SBATCH --error=ddpm%j.err  # error file name
#SBATCH --qos=qos_gpu-t4
#SBATCH --partition=gpu_p4

source /gpfswork/rech/gft/umh25bv/miniconda3/bin/activate /gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv

/gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv/bin/python3 -u /gpfswork/rech/gft/umh25bv/conditional_ddpm/main.py \
   --mode train --dataset dataset_rh-jeanzay --labels pipelines \
   --batch_size 4 --data_dir data \
   --n_epoch 100 --lrate 1e-4 --sample_dir samples