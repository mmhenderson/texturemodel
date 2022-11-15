#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
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
cd ${ROOT}texturemodel/code/feature_extraction

debug=0
use_node_storage=0
which_prf_grid=5
padding_mode='reflect'
type=alexnet
min_pct_var=95
zscore=0
max_pc_to_retain=100
which_prf_grid=5

subjects=(1 2 3 4 5 6 7 8)

for subject in ${subjects[@]}
do

    python3 extract_alexnet_features.py --subject $subject --use_node_storage $use_node_storage --debug $debug --which_prf_grid $which_prf_grid --padding_mode $padding_mode
    
    python3 pca_feats.py --subject $subject --debug $debug --type $type --zscore $zscore --max_pc_to_retain $max_pc_to_retain --min_pct_var $min_pct_var --which_prf_grid $which_prf_grid
    
    fn=${ROOT}features/alexnet/S${subject}_Conv1_ReLU_reflect_features_each_prf_grid${which_prf_grid}.h5py
    echo Removing $fn
    rm $fn
    fn=${ROOT}features/alexnet/S${subject}_Conv2_ReLU_reflect_features_each_prf_grid${which_prf_grid}.h5py
    echo Removing $fn
    rm $fn
    fn=${ROOT}features/alexnet/S${subject}_Conv3_ReLU_reflect_features_each_prf_grid${which_prf_grid}.h5py
    echo Removing $fn
    rm $fn
    fn=${ROOT}features/alexnet/S${subject}_Conv4_ReLU_reflect_features_each_prf_grid${which_prf_grid}.h5py
    echo Removing $fn
    rm $fn
    fn=${ROOT}features/alexnet/S${subject}_Conv5_ReLU_reflect_features_each_prf_grid${which_prf_grid}.h5py
    echo Removing $fn
    rm $fn
    
    
done

