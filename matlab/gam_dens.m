function ng = gam_dens(shape, scale, data)
% This code was ported from the nipy project (http://nipy.org) and was
% originally contrributed by Bertrand Thirion
    ng = zeros(size(data));
    cst = - shape * log(scale) - gammaln(shape);
    i = data > 0;
    if ~isempty(i)
        lz = cst + (shape - 1) * log(data(i)) - data(i) / scale;
        ng(i) = exp(lz);
    end
end