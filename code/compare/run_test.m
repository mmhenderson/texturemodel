function run_test()

    % COMPARE TEXTURE STATISTICS HERE TO MY VERSION OF THIS CODE IN PYTHON.
    % Will print several examples of numerical value for each quantity, so that we can directly
    % verify the computations are the same. 
    % Compare to 'pyramid_texture_model_CompareToMatlab.ipynb'
    
    root = pwd;
    
    path = fullfile(root,'metamers');   
    fprintf('adding %s to path\n',path);
    addpath(genpath(path));
    
    path = fullfile(root,'matlabPyrTools');
    fprintf('adding %s to path\n',path);
    addpath(genpath(path));


    % load a test image
    oim = double(imread('example-im-512x512.png'));

    % set options
    % opts = metamerOpts(oim,'windowType=square','nSquares=[4 4]');
    opts = metamerOpts(oim,'windowType=radial', 'aspect=1','scale=1');
    opts.outputPath = pwd;

    % generate the masks structure (will over-write some of this)
    m = mkImMasks(opts);

    na_to_use = [7,7,5,3,3]
    for sc=1:5
  
        % Manually setting parameters for one of the spatial 'masks' to match the parameters for one of my pRFs.
        % corresponding prf params are (x,y,sigma) = (0,0,0.5)
        % Set the bounding box, and size of the autocorrelation matrix to retain.
        % these match the values i'm using in my python version of this code. Could use anything 
        % for the mask, but this will make it easy to compare.     
        % Note this means that only the results for the first mask will match the python code.
        % The other masks will give other values.
        
        npix = size(m.scale{sc}.maskMat, 2);      
        m.scale{sc}.Na(1) = na_to_use(sc);
        m.scale{sc}.ind(1,:) = [1, npix, 1, npix];
        
        % Create the fake gaussian
        gauss = quick_gaussian(npix);
        gauss = double(gauss);
        
        %[xvals, yvals] = meshgrid(linspace(-1,1,npix), linspace(-1,1,npix));
        %rvals = sqrt(xvals.^2 + yvals.^2);
        %gauss = exp(-rvals/0.5); 
        
        m.scale{sc}.maskMat = permute(repmat(gauss, 1,1, size(m.scale{sc}.maskMat, 1)),[3,1,2]);
        m.scale{sc}.maskNorm = zeros(size(m.scale{sc}.maskNorm));
        for mm=1:size(m.scale{sc}.maskNorm,2)

            norm = m.scale{sc}.maskMat(mm,:,:);
            norm = norm(:)/sum(norm(:));
            m.scale{sc}.maskNorm(:,mm) = norm;

        end
    end

    % do metamer analysis on test image (this fn will print a bunch of stuff at the end)
    params = metamerAnalysis_TEST(oim,m,opts);
    
    function gauss = quick_gaussian(npix)

        %size = 1.0;
        x=0;
        y=0;
        sigma=0.5;
        aperture=1.0;
        
        % first meshgrid over image space
        [xgrid,ygrid] = meshgrid(linspace(-aperture/2, aperture/2, npix), ...
                      linspace(-aperture/2, aperture/2, npix));
        x_centered = xgrid-x;
        y_centered = ygrid-(-y);
    
        gauss = exp(-(x_centered.^2 + y_centered.^2)/(2*sigma^2));
    
        % normalize so it will sum to 1
        gauss = gauss./sum(gauss(:));


        %deg = size;
        %dpix = single(deg) / npix;
        %pix_min = -deg/2.0 + 0.5 * dpix;
        %pix_max = deg/2.0;
        %[Xm, Ym] = meshgrid([pix_min:dpix:pix_max], [pix_min:dpix:pix_max]);

        %d = (2*single(sigma)^2);
        %A = single(1. / (d*pi));
        %Zm = dpix.^2 * A * exp(-((Xm-x).^2 + (-Ym-y).^2)./ d);
        %gauss = single(Zm); *)

        return 
        
    return