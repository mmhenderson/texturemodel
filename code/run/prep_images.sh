#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

source ~/myenv/bin/activate

# change this path
ROOT=/user_data/mmhender/

# put the code directory on your python path
PYTHONPATH=:${ROOT}texturemodel/code/${PYTHONPATH}
echo $PYTHONPATH

# go to folder where script is located
cd ${ROOT}texturemodel/code/utils/

# to test the code, use debug=1
# to run for real, set debug=0 (False)
debug=1

python3 prep_images.py --debug $debug
