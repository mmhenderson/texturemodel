#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --exclude=mind-1-23,mind-1-34
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=12-00:00:00

set -e

echo $SLURM_JOBID
echo $SLURM_NODELIST

source ~/myenv/bin/activate

module load matlab-9.7

gist_code_dir=/user_data/mmhender/texturemodel/code/feature_extraction/gist_matlab/
cd ${gist_code_dir}
gist_code_dir='"'${gist_code_dir}'"'


image_dir=/user_data/mmhender/nsd/stimuli/

save_dir=/user_data/mmhender/features/gist/
save_dir='"'${save_dir}'"'

# subject_list=(1)
subject_list=(1 2 3 4 5 6 7 8)
debug=0
overwrite=1

for nsd_subject_id in ${subject_list[@]}
do

    # path where NSD image brick is located
    images_filename='"'${image_dir}S${nsd_subject_id}_stimuli_240.h5py'"'
    
    matlab -nodisplay -nodesktop -nosplash -r "get_gist($nsd_subject_id,$images_filename,$save_dir,$overwrite,$debug); exit"

done
