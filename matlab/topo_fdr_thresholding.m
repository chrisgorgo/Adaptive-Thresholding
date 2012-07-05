function topo_fdr_thresholding(spm_mat_file, con_index, cluster_forming_thr, thresDesc, use_topo_fdr, force_activation, cluster_extent_p_fdr_thr, stat_filename, height_threshold_type, extent_threshold)

    % con_index = 2;
    % cluster_forming_thr = 2.771290;
    % thresDesc  = 'none';
    % use_topo_fdr  = 1;
    % force_activation  = 1;
    % cluster_extent_p_fdr_thr = 0.050000;
    % stat_filename = '/media/chris_filo_backu/2010reliability/workdir_fmri/group/first_level/main/model/threshold_topo_ggmm/_session_first/_subject_id_3a3e1a6f-dc92-412c-870a-74e4f4e85ddb/_partition_all/_task_name_finger_foot_lips/topo_fdr/mapflow/_topo_fdr1/foot_t_map.img';
    % height_threshold_type = 'stat';
    % extent_threshold = 0;
    load(spm_mat_file);

    FWHM  = SPM.xVol.FWHM;
    df = [SPM.xCon(con_index).eidf SPM.xX.erdf];
    STAT = SPM.xCon(con_index).STAT;
    R = SPM.xVol.R;
    S = SPM.xVol.S;
    n = 1;

    switch thresDesc
        case 'FWE'
            cluster_forming_thr = spm_uc(cluster_forming_thr,df,STAT,R,n,S);

        case 'none'
            if strcmp(height_threshold_type, 'p-value')
                cluster_forming_thr = spm_u(cluster_forming_thr^(1/n),df,STAT);
            end
    end

    stat_map_vol = spm_vol(stat_filename);
    [stat_map_data, stat_map_XYZmm] = spm_read_vols(stat_map_vol);

    Z = stat_map_data(:)';
    [x,y,z] = ind2sub(size(stat_map_data),(1:numel(stat_map_data))');
    XYZ = cat(1, x', y', z');

    XYZth = XYZ(:, Z >= cluster_forming_thr);
    Zth = Z(Z >= cluster_forming_thr);
    [pathstr, name, ext] = fileparts(stat_filename)
    spm_write_filtered(Zth,XYZth,stat_map_vol.dim',stat_map_vol.mat,'thresholded map', strrep(stat_filename,ext, '_pre_topo_thr.hdr'));

    max_size = 0;
    max_size_index = 0;
    th_nclusters = 0;
    nclusters = 0;
    if isempty(XYZth)
        thresholded_XYZ = [];
        thresholded_Z = [];
    else
        if use_topo_fdr
            V2R        = 1/prod(FWHM(stat_map_vol.dim > 1));
            [uc,Pc,ue] = spm_uc_clusterFDR(cluster_extent_p_fdr_thr,df,STAT,R,n,Z,XYZ,V2R,cluster_forming_thr);
        end

        voxel_labels = spm_clusters(XYZth);
        nclusters = max(voxel_labels);

        thresholded_XYZ = [];
        thresholded_Z = [];

        for i = 1:nclusters
            cluster_size = sum(voxel_labels==i);
             if cluster_size > extent_threshold && (~use_topo_fdr || (cluster_size - uc) > -1)
                thresholded_XYZ = cat(2, thresholded_XYZ, XYZth(:,voxel_labels == i));
                thresholded_Z = cat(2, thresholded_Z, Zth(voxel_labels == i));
                th_nclusters = th_nclusters + 1;
             end
            if force_activation
                cluster_sum = sum(Zth(voxel_labels == i));
                if cluster_sum > max_size
                    max_size = cluster_sum;
                    max_size_index = i;
                end
            end
        end
    end

    activation_forced = 0;
    if isempty(thresholded_XYZ)
        if force_activation && max_size ~= 0
            thresholded_XYZ = XYZth(:,voxel_labels == max_size_index);
            thresholded_Z = Zth(voxel_labels == max_size_index);
            th_nclusters = 1;
            activation_forced = 1;
        else
            thresholded_Z = [0];
            thresholded_XYZ = [1 1 1]';
            th_nclusters = 0;
        end
    end

    fprintf('activation_forced = %d\n',activation_forced);
    fprintf('pre_topo_n_clusters = %d\n',nclusters);
    fprintf('n_clusters = %d\n',th_nclusters);
    fprintf('cluster_forming_thr = %f\n',cluster_forming_thr);

    spm_write_filtered(thresholded_Z,thresholded_XYZ,stat_map_vol.dim',stat_map_vol.mat,'thresholded map', strrep(stat_filename,ext, '_thr.hdr'));

end

