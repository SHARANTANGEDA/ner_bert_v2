#!/bin/bash
#SBATCH -p gpu
#SBATCH  -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 32G
#SBATCH -t 1-00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name="range_test"
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-user=f20170515@hyderabad.bits-pilani.ac.in
#SBATCH --mail-type=NONE

module purge
module load python-3.8.6-gcc-10.2.0-csfajta
#module load py-six-1.14.0-gcc-10.2.0-dgubhyv
module load gcc-10.2.0-gcc-8.3.1-25ppgbv
#module load cuda-11.1.1-gcc-10.2.0-hmrghph
module load cuda-11.0.2-gcc-10.2.0-3wlbq6u # Use CUDA 11.0 for safety
module load libpng-1.6.37-gcc-10.2.0-iqar3xu
module load openblas-0.3.13-gcc-10.2.0-gws6lqm
module load openmpi-4.0.5-gcc-10.2.0-vx4yhsi
#module load cudnn-8.0.4.30-11.1-gcc-10.2.0-6wjnurb
module load cudnn-8.0.4.30-11.0-gcc-10.2.0-wilsmzv # Use cuDNN 8.0.4 with CUDA 11.0

source ~/mlenv/bin/activate

cd ~/ner_bert_v2

pipenv install
pipenv shell
export $(xargs < envs/local.env)
python train_validate