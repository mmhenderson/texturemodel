import os, sys, subprocess

# updating files in this folder based on any updates i made to the original code

sourcepath = '/user_data/mmhender/modfit/code/'

destpath = '/user_data/mmhender/texturemodel/code/'



folder = 'model_fitting'
exclude_names = ['semantic_discrim_raw.py','feature_corrs_raw.py', 'subsample_trials.py']
files = os.listdir(os.path.join(sourcepath, folder))
files = [fi for fi in files if fi[-3:]=='.py' and fi not in exclude_names]
print(files)

for fi in files:
    
    file1 = os.path.join(sourcepath, folder, fi)
    file2 = os.path.join(destpath, folder)

    if not os.path.exists(file2):
        os.makedirs(file2)
        
    cmd = ['rsync', file1, file2] 
    print(cmd)
    err = subprocess.call(cmd)
    print(err)

    
    
    
folder = 'feature_extraction'
include_names = ['fwrf_features.py', 'merge_features.py','pca_feats.py', \
                 'extract_pyramid_texture_features.py', 'texture_statistics_pyramid.py', 
                 'extract_alexnet_features.py', 'extract_clip_features.py', \
                 'semantic_features.py', 'texture_feature_utils.py', '__init__.py']

files = include_names

for fi in files:
    
    file1 = os.path.join(sourcepath, folder, fi)
    file2 = os.path.join(destpath, folder)

    if not os.path.exists(file2):
        os.makedirs(file2)
        
    cmd = ['rsync', file1, file2] 
    print(cmd)
    err = subprocess.call(cmd)
    print(err)


    
    
folder = 'plotting'
exclude_names = []
files = os.listdir(os.path.join(sourcepath, folder))
files = [fi for fi in files if fi[-3:]=='.py' and fi not in exclude_names]
print(files)

for fi in files:
    
    file1 = os.path.join(sourcepath, folder, fi)
    file2 = os.path.join(destpath, folder)

    if not os.path.exists(file2):
        os.makedirs(file2)
        
    cmd = ['rsync', file1, file2] 
    print(cmd)
    err = subprocess.call(cmd)
    print(err)


    
    
folder = 'utils'
exclude_names = []
files = os.listdir(os.path.join(sourcepath, folder))
files = [fi for fi in files if fi[-3:]=='.py' and fi not in exclude_names]
print(files)

for fi in files:
    
    file1 = os.path.join(sourcepath, folder, fi)
    file2 = os.path.join(destpath, folder)

    if not os.path.exists(file2):
        os.makedirs(file2)
        
    cmd = ['rsync', file1, file2] 
    print(cmd)
    err = subprocess.call(cmd)
    print(err)
    
   
    
folder = 'run'
include_names = ['fit_pyramid_big.sh']

files = include_names

for fi in files:
    
    file1 = os.path.join(sourcepath, folder, fi)
    file2 = os.path.join(destpath, folder)

    if not os.path.exists(file2):
        os.makedirs(file2)
        
    cmd = ['rsync', file1, file2] 
    print(cmd)
    err = subprocess.call(cmd)
    print(err)


   