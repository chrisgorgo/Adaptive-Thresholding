function threshold = ggmm_thresholding(stat_filename, mask_filename)
    mask_data = spm_read_vols(spm_vol(mask_filename));
    stat_data = spm_read_vols(spm_vol(stat_filename));
    stat_data = stat_data(mask_data > 0);
    no_signal_components = {Gaussian(0, 10)};
    noise_and_activation_components = {Gaussian(0, 10), Gamma(4, 5, 0)};
    noise_activation_and_deactivation_components = {NegativeGamma(4,5, 0), Gaussian(0, 10), Gamma(4, 5, 0)};

    models = {no_signal_components, noise_and_activation_components, noise_activation_and_deactivation_components};
    best_bic = inf;

    for i=1:length(models)
        em = ExpectationMaximization(models{i});
        em.fit(stat_data);
        em.plot(stat_data);
        bic = em.BIC(stat_data);
        if bic < best_bic
            best_bic = bic;
            best_model = em;
        end
    end

    if size(best_model.components,2) == 1
        fprintf('No signal found!\n')
        return
    elseif size(best_model.components,2) == 2
        pp = best_model.posteriors(stat_data);
        active_map = (pp(:, 2) > pp(:, 1)) & stat_data > 0.01;
    elseif size(best_model.components,2) == 3
        pp = best_model.posteriors(stat_data);
        active_map = (pp(:, 3) > pp(:, 2)) & stat_data > 0.01;
    end

    threshold = min(stat_data(active_map));

    if length(threshold) == 0
        fprintf('No signal found!\n')
        return
    end
end