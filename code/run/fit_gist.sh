#!/bin/bash
#SBATCH --partition=tarrq
#SBATCH --gres=gpu:1
#SBATCH --exclude=mind-1-23
#SBATCH --mem=32G
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

# go to folder where script is located
cd ${ROOT}texturemodel/code/model_fitting

subjects=(1 2 3 4 5 6 7 8)

debug=0
up_to_sess=40
# debug=1
# up_to_sess=1

use_precomputed_prfs=0
which_prf_grid=0

from_scratch=1
date_str=0
overwrite_sem_disc=1
do_val=1
do_tuning=0
do_sem_disc=0

fitting_type=gist
n_ori_gist=4 
n_blocks_gist=2

for subject in ${subjects[@]}
do
   
    python3 fit_model.py --subject $subject --debug $debug --up_to_sess $up_to_sess --use_precomputed_prfs $use_precomputed_prfs --which_prf_grid $which_prf_grid --from_scratch $from_scratch --date_str $date_str --do_val $do_val --do_tuning $do_tuning --do_sem_disc $do_sem_disc --overwrite_sem_disc $overwrite_sem_disc --fitting_type $fitting_type --n_ori_gist $n_ori_gist --n_blocks_gist $n_blocks_gist
    
done
