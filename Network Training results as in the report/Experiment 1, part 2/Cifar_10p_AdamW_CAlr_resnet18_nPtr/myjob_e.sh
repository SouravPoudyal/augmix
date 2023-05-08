#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=e
#SBATCH --output=e.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sourav.poudyal@student.uni-siegen.de
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=35GB
#SBATCH --gres=gpu:1

module load GpuModules
module load pytorch-extra-py37-cuda11.2-gcc8/1.9.1

python cifar_t2_e.py -m resnet18 -o AdamW -s CosineAnnealingLR
