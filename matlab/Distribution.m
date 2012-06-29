classdef Distribution < handle
    properties(Constant,Abstract)
        free_parameters
    end
    methods(Abstract)
        fit_weighted(obj, data, weights);
        pdf(obj, data);
    end 
end

