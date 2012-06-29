function adaptive_thresholding(stat_filename, mask_filename, spm_mat_file, con_index)
    thr = ggmm_thresholding(stat_filename, mask_filename);
    topo_fdr_thresholding(spm_mat_file, con_index, thr, 'none', 1, 1, 0.05, stat_filename, 'stat', 0);
end