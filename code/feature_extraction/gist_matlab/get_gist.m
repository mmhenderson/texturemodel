function get_gist(nsd_subject_id, images_filename, save_dir, overwrite, debug)

    if nargin==0
        nsd_subject_id = 1;
        images_filename = sprintf('/user_data/mmhender/nsd/stimuli/S%d_stimuli_240.h5py', nsd_subject_id);
        save_dir = '/user_data/mmhender/features/gist/';
        debug=1;
        overwrite=0;
    end
    
    if debug==0
        batch_size=10000;
    else
        batch_size=10;
    end
    
    fprintf('Reading image brick from %s...\n', images_filename)
    I = h5read(images_filename, '/stimuli');
    [h, w, c, n_total_ims] = size(I);
    fprintf('size of image brick:\n')
    disp([h,w,c,n_total_ims])

    I_batch = I(:,:,:,1:batch_size);
 
    param.imageSize = 128;
    %param.orientationsPerScale = [8 8 8 8];
    param.orientationsPerScale = [4 4 4 4];
    %param.numberBlocks = 4;
    param.numberBlocks = 2;
    param.fc_prefilt = 4;
    
    [gist, param] = LMgist(I_batch, 0, param);

    fprintf('size of gist output:\n')
    disp(size(gist))
    disp(param)
    
    if ~exist(save_dir, 'dir')
       mkdir(save_dir)
    end
    
    %save_gist_filename = fullfile(save_dir, sprintf('S%d_gistdescriptors_8ori.mat', nsd_subject_id));
    %save_gist_filename = fullfile(save_dir, sprintf('S%d_gistdescriptors_4ori.mat', nsd_subject_id));
    save_gist_filename = fullfile(save_dir, sprintf('S%d_gistdescriptors_4ori_2blocks.mat', nsd_subject_id));
    fprintf('saving to %s\n', save_gist_filename);
    save(save_gist_filename, 'gist','param');
    
    % put into the format we will need later, which will be [n_images x n_features x n_prfs]
    % (n_prfs is just 1 here, because the features are entire image)
    % the dims get reversed from matlab to python, so needs to be opposite order
    [n_images, n_features] = size(gist);
    gist_reshaped = reshape(gist', [1,n_features, n_images]);
    
    %save_gist_filename_h5 = fullfile(save_dir, sprintf('S%d_gistdescriptors_8ori.h5py', nsd_subject_id));
    %save_gist_filename_h5 = fullfile(save_dir, sprintf('S%d_gistdescriptors_4ori.h5py', nsd_subject_id));
    save_gist_filename_h5 = fullfile(save_dir, sprintf('S%d_gistdescriptors_4ori_2blocks.h5py', nsd_subject_id));
    if exist(save_gist_filename_h5,'file')
        if overwrite
            fprintf('File exists, removing it now...\n')
            unix(char(sprintf('rm %s',save_gist_filename_h5)));
        else
            error('File already exists. To overwrite it set overwrite=True.');
        end
    end
    fprintf('saving to %s\n', save_gist_filename_h5);
    h5create(save_gist_filename_h5,'/features',[1, n_features, n_images]);
    h5write(save_gist_filename_h5, '/features', gist_reshaped, [1,1,1], size(gist_reshaped));


end




