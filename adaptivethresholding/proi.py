'''
Created on 04.04.2013

@author: filo
'''
from adaptivethresholding.gamma_fit import GaussianComponent,\
    FixedMeanGaussianComponent, GammaComponent, NegativeGammaComponent,\
    RestrictedPositiveGaussianComponent, EM
    
import numpy as np

def threshold(stat_map, prior_map):
    #fit mixture model
    em = EM([NegativeGammaComponent(4, 5), GaussianComponent(0,1), GammaComponent(4, 5)], weights=prior_map)
    print "fitting the mixture"
    em.fit(stat_map)

    roi_posteriors = em.posteriors(stat_map.flatten())
    roi_posteriors[np.isnan(roi_posteriors)] = 0
    
    #fit gaussian
    gaussian = GaussianComponent(0,1)
    print "fitting the gaussian"
    gaussian.fit_weighted(stat_map, 1-prior_map)
    nonroi_posterior = gaussian.pdf(stat_map)*(1-prior_map)
    nonroi_posterior[np.isnan(nonroi_posterior)] = 0
    
    #roi or not?
    
    in_roi = roi_posteriors.sum(axis=1) > nonroi_posterior.flatten()
    
    thresholded_map = np.zeros(stat_map.shape)
    thresholded_map[in_roi.reshape(stat_map.shape)] = roi_posteriors[:,1][in_roi] < roi_posteriors[:,2][in_roi]
    
    return thresholded_map.reshape(stat_map.shape), em, gaussian