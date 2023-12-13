#!/bin/bash
#SBATCH --job-name=ddpm-4class # nom du job
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=16           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=72:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=ddpm%j.out # output file name
#SBATCH --error=ddpm%j.err  # error file name
#SBATCH --qos=qos_gpu-t4

source /gpfswork/rech/gft/umh25bv/miniconda3/bin/activate /gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv

python -u main.py --mode transfer --dataset dataset_rh_4class-jeanzay \
--labels pipelines    --batch_size 1 --data_dir data --sample_dir samples-4classes \
--save_dir models-4classes --test_iter 90 --n_classes 4

# /gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv/bin/python3 -u /gpfswork/rech/gft/umh25bv/conditional_ddpm/main.py \
#    --mode transfer --dataset dataset_rh-jeanzay --labels pipelines \
#    --batch_size 1 --data_dir data \
#    --sample_dir samples --save_dir models \
#    --test_iter 90 --n_classes 24

# /gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv/bin/python3 -u /gpfswork/rech/gft/umh25bv/conditional_ddpm/main.py \
#    --mode train --dataset dataset_rh_4class-jeanzay --labels pipelines \
#    --batch_size 8 --data_dir data --n_classes 4\
#    --n_epoch 500 --lrate 1e-4 --sample_dir samples-4classes --save_dir models-4classes

# /gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv/bin/python3 -u /gpfswork/rech/gft/umh25bv/conditional_ddpm/main.py \
#    --mode train --dataset dataset_rh-jeanzay --labels pipelines \
#    --batch_size 8 --data_dir data \
#    --n_epoch 100 --lrate 1e-4 --sample_dir samples --save_dir models