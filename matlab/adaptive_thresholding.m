function adaptive_thresholding(stat_filename, mask_filename, spm_mat_file, con_index)

% read the data (spmT and mask), call the EM code and return the T value
% for the cluster forming threshold
%
% FORMAT: [threshold,em,bic] = ggmm_thresholding(stat_filename, mask_filename)
%
% INPUT: stat_filename: full name (ie with path) of the spmT image
%        mask_filename: full name (ie with path) of the mask image
% 
% OUTPUT: threshold: NaN if Gaussian only was found best model
%                    one value X if positive Gamma was found best model,
%                    two values X1 X2 if negative and positive Gamma was found best model,
%         em: object that contains the models 
%         bic: Bayesian information criteria of each model
%
% ------------------------------------------
% Chris Gorgolewski 29 June 2012


[threshold,bic] = ggmm_thresholding(stat_filename, mask_filename);
topo_fdr_thresholding(spm_mat_file, con_index, thr, 'none', 1, 1, 0.05, stat_filename, 'stat', 0);
