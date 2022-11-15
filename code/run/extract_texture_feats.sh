#!/bin/bash
#SBATCH --partition=tarrq
#SBATCH --gres=gpu:1
#SBATCH --exclude=mind-1-23
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

echo $SLURM_JOBID
echo $SLURM_NODELIST

source ~/myenv/bin/activate

# change this path
ROOT=/user_data/mmhender/

# put the code directory on your python path
PYTHONPATH=:${ROOT}texturemodel/code/${PYTHONPATH}
echo $PYTHONPATH

# go to folder where script is located
cd ${ROOT}texturemodel/code/feature_extraction

debug=0

use_node_storage=0
n_ori=4
n_sf=4
batch_size=50
which_prf_grid=5

sublist=(1 2 3 4 5 6 7 8)
# sublist=(1)

for subj in ${sublist[@]}
do

    python3 extract_pyramid_texture_features.py --subject $subj --use_node_storage $use_node_storage --n_ori=$n_ori --n_sf=$n_sf --batch_size $batch_size --debug $debug --which_prf_grid $which_prf_grid 

done

max_pc_to_retain=100
min_pct_var=95
pca_type=pcaHL
save_weights=1

for subject in ${sublist[@]}
do
    
    python3 pca_texture_feats.py --subject $subject --debug $debug --max_pc_to_retain $max_pc_to_retain --min_pct_var $min_pct_var --pca_type $pca_type --which_prf_grid $which_prf_grid --save_weights $save_weights 

done