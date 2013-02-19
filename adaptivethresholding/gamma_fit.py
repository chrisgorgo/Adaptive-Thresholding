'''
Created on 14 May 2010

@author: filo
'''
from scipy.stats.distributions import gamma, norm

import numpy as np
from scipy.optimize import fmin
import math
from nipy.algorithms.clustering.ggmixture import _gam_param
import pylab as plt
from copy import deepcopy

def opt_func(param, x, resp):
    return -np.sum(np.log(gamma.pdf(x, param[0], scale=param[1])) * resp, axis=0)

def fit_norm(data, shape, scale, resp):
    return fmin(opt_func, (shape, scale), args=(data, resp), disp=True)

def calc_resp(data, shape1, scale1, shape2, scale2):
    resp1 = gamma.pdf(data, shape1, scale=scale1)
    resp2 = gamma.pdf(data, shape2, scale=scale2)
    return resp1, resp2

class GammaComponent(object):

    free_parameters = 2

    def __init__(self, shape, scale):
        self.shape = shape
        self.scale = scale
        self.mu = 0

    def pdf(self, data):
        return gamma.pdf(data - self.mu, self.shape, scale=self.scale)

    def fit_weighted(self, data, weights):

        (self.shape, self.scale) = _gam_param(data - self.mu, weights)

class RestrictedPositiveGaussianComponent(object):

    free_parameters = 2

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.limit = 0

    def pdf(self, data):
        return norm.pdf(data - self.limit, loc=self.mu, scale=math.sqrt(self.sigma))

    def fit_weighted(self, data, weights):
        weights[data < self.limit] = 0
        weights_sum = weights.sum()
        self.mu = np.sum((data - self.limit) * weights) / weights_sum
        self.sigma = np.sum(((data - self.limit) - self.mu) ** 2 * weights) / weights_sum

class RestrictedNegativeGaussianComponent(RestrictedPositiveGaussianComponent):

    def fit_weighted(self, data, weights):
        weights[data > self.limit] = 0
        weights_sum = weights.sum()
        self.mu = np.sum((data - self.limit) * weights) / weights_sum
        self.sigma = np.sum(((data - self.limit) - self.mu) ** 2 * weights) / weights_sum

class NegativeGammaComponent(object):

    free_parameters = 2

    def __init__(self, shape, scale):
        self.shape = shape
        self.scale = scale
        self.mu = 0

    def pdf(self, data):
        return gamma.pdf(-(data - self.mu), self.shape, scale=self.scale)

    def fit_weighted(self, data, weights):

        (self.shape, self.scale) = _gam_param(-(data - self.mu), weights)
        #print (self.shape, self.scale)

class GaussianComponent(object):

    free_parameters = 2

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def pdf(self, data):
        #print self.sigma
        return norm.pdf(data, loc=self.mu, scale=math.sqrt(self.sigma))

    def fit_weighted(self, data, weights):
        weights_sum = weights.sum()
        self.mu = np.sum(data * weights) / weights_sum
        self.sigma = np.sum((data - self.mu) ** 2 * weights) / weights_sum

class FixedMeanGaussianComponent(object):

    free_parameters = 1

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def pdf(self, data):
        #print self.sigma
        return norm.pdf(data, loc=self.mu, scale=math.sqrt(self.sigma))

    def fit_weighted(self, data, weights):
        weights_sum = weights.sum()
        #self.mu = np.sum(data*weights)/weights_sum
        self.sigma = np.sum((data - self.mu) ** 2 * weights) / weights_sum

class PositiveGaussianComponent(object):

    free_parameters = 2

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def pdf(self, data):
        #print self.sigma
        return norm.pdf(data, loc=self.mu, scale=math.sqrt(self.sigma))

    def fit_weighted(self, data, weights):
        data = data[data > 0]
        weights = weights[data > 0]
        weights_sum = weights.sum()
        self.mu = np.sum(data * weights) / weights_sum
        self.sigma = np.sum((data - self.mu) ** 2 * weights) / weights_sum

class NegativeGaussianComponent(object):

    free_parameters = 2

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def pdf(self, data):
        #print self.sigma
        return norm.pdf(data, loc=self.mu, scale=math.sqrt(self.sigma))

    def fit_weighted(self, data, weights):
        data = data[data < 0]
        weights = weights[data < 0]
        weights_sum = weights.sum()
        self.mu = np.sum(data * weights) / weights_sum
        self.sigma = np.sum((data - self.mu) ** 2 * weights) / weights_sum

class EM(object):

    def __init__(self, components, mix_type="multi"):
        self.components = components
        self._mix_type = mix_type


    def _E(self, data):
            resp = self.posteriors(data)
            resp_sum = resp.sum(axis=1)
            resp = resp / np.tile(resp_sum.reshape(-1, 1), len(self.components))
            return resp

    def _M(self, data, resp):
            for i, component in enumerate(self.components):
                if (isinstance(component, NegativeGammaComponent) or isinstance(component, GammaComponent)):
                    for c in self.components:
                        if isinstance(c, GaussianComponent):
                            component.mu = c.mu
                            break
                if (isinstance(component, RestrictedNegativeGaussianComponent) or isinstance(component, RestrictedPositiveGaussianComponent)):
                    for c in self.components:
                        if isinstance(c, GaussianComponent):
                            component.limit = c.mu
                            break

                component.fit_weighted(data, resp[:, i])
                if self._mix_type == "multi":
                    self.mix[:,i] = resp[:, i]
                else: 
                    self.mix[i] = resp[:, i].sum() / resp[:, i].size
#                print self.mix[i]

    def loglikelihood(self, data):
        likelihood = self.posteriors(data)
        sum_likelihood = likelihood.sum(axis=1)

        return np.sum(np.log(sum_likelihood))

    def posteriors(self, data):
        posteriors = np.zeros((data.size, len(self.components)))
        for i, component in enumerate(self.components):
            posteriors[:, i] = component.pdf(data)
        posteriors[np.isnan(posteriors)] = 0
        posteriors = posteriors * self.mix
        return posteriors

    def BIC(self, data):
        k = 0
        for component in self.components:
            k += component.free_parameters
        return -2.*self.loglikelihood(data) + k * np.log(data.size)

    def fit(self, data, maxiter=100, min_improvement=1.e-4):
        if self._mix_type == "multi":
            self.mix = np.ones((len(data),len(self.components))) * 1 / len(self.components)
        else:
            self.mix = np.ones(len(self.components)) * 1 / len(self.components)
        prev_loglike = 0

#        xRange = np.arange(math.floor(min(data)), math.ceil(max(data)), 0.1)

        for i in range(maxiter):

#            plt.figure()
#            plt.hist(data, bins=50, normed=True)

#            pdf_sum = np.zeros(xRange.shape)
#            for j, component in enumerate(self.components):
#                pdf = component.pdf(xRange)#*self.mix[j]
#                plt.plot(xRange, pdf)
#                pdf_sum += pdf
#            plt.plot(xRange, pdf_sum)

#    
            resp = self._E(data)
            #print resp.sum(axis=1).min()
            self._M(data, resp)

            cur_loglike = self.loglikelihood(data)
            improv = cur_loglike - prev_loglike
            if i != 0 and improv <= min_improvement:
                print "break %f" % improv
                break
            prev_loglike = cur_loglike
#            if i == 65:
#                plt.show()
            print "log likelihood = %f, improv = %f, iter = %d" % (self.loglikelihood(data), improv, i)
#        plt.show()

if __name__ == '__main__':
#            
#    random_sample1 = norm.rvs(loc=0, scale=10, size = 5000)
#    random_sample2 = gamma.rvs(3, scale = 2, size = 50)
#    random_sample3 = -gamma.rvs(3, scale = 2, size = 50)
#    
#    random_sample = np.concatenate((random_sample1, random_sample2, random_sample3), axis=0)
    import pylab as plt
    import nibabel as nb
    #img = nb.load("/home/filo/workspace/ROIThresholding/workingdir/main_workflow/_SNR_0.5_activation_shape_[96, 96, 1]_sim_id_16/contrastestimate/spmT_0001.hdr")
    #img = nb.load("/home/filo/data/feeds/examples/fmri.feat/stats/tstat1.nii.gz")
    #img = nb.load("/home/filo/workspace/fmri_tumour/output/pilot1/contrasts/finger_foot_lipsBet_fsl/_subject_id_pilot1/spmT_0002.hdr")
    #img = nb.load("/home/filo/workspace/fmri_tumour/output/pilot1/contrasts/finger_tappingBet_fsl/_subject_id_pilot1/spmT_0001.hdr")
    #img = nb.load("/home/filo/workspace/nipype/examples/spm_auditory_tutorial/workingdir/level1/firstlevel/analysis/_subject_id_M00223/contrastestimate/spmT_0001.hdr")
    #img = nb.load("/home/filo/workspace/nipype/examples/fsl_feeds/workingdir/level1/firstlevel/modelfit/_subject_id_data/modelestimate/results/tstat1.nii.gz")
    #img = nb.load("/home/filo/workspace/nipype/examples/spm_tutorial2/l1output/s1/contrasts/_subject_id_s1/_fwhm_4/spmT_0001.hdr")
    #img = nb.load("/home/filo/workspace/nipype/examples/spm_face_tutorial/workingdir/level1/firstlevel/analysis/_subject_id_M03953/contrastestimate/spmT_0001.hdr")
    #img = nb.load("/home/filo/workspace/ROIThresholding/workingdir/_SNR_0.5_activation_shape_[32, 32, 1]_sim_id_7/contrastestimate/spmT_0001.hdr")
    #img = nb.load("/mnt/data/case_studies/workdir_fmri/17904/pipeline/functional_run/model/threshold_topo_ggmm/_task_name_finger_foot_lips/ggmm/mapflow/_ggmm4/foot_vs_other_t_map.img")
    img = nb.load("/home/filo/workspace/Adaptive-Thresholding/matlab/data/dset2/spmT_0003.img")
    #img = nb.load("/home/filo/workspace/ROIThresholding/spm_face_tutorial/workingdir/level1/firstlevel/masked_analysis/_subject_id_M03953/contrastestimate/spmT_0001.hdr")
    #img = nb.load("/home/filo/workspace/ROIThresholding/spm_face_tutorial/workingdir/level1/firstlevel/analysis/_subject_id_M03953/contrastestimate/spmT_0001.hdr")
    #mask = nb.load("/mnt/data/case_studies/workdir_fmri/17904/pipeline/functional_run/model/_task_name_finger_foot_lips/level1estimate/mask.img")
    mask = nb.load('/home/filo/workspace/Adaptive-Thresholding/matlab/data/dset2/mask.img')
    mask = mask.get_data().reshape(-1,1,order='F').copy()
    data = img.get_data()
    data = data.reshape(-1,1,order='F').copy()
    data = data[mask > 0]
    #data = np.array(img.get_data().ravel())

    components = []

    #components.append(GaussianComponent(5, 10))
    #components.append(GaussianComponent(-5, 10))
    no_signal_components = [GaussianComponent(0, 1)]
    no_signal_zero_components = [FixedMeanGaussianComponent(0, 10)]

    noise_and_activation_components = deepcopy(no_signal_components)
    noise_and_activation_components.append(GammaComponent(4, 5))
    noise_zero_and_activation_components = deepcopy(no_signal_zero_components)
    noise_zero_and_activation_components.append(GammaComponent(4, 5))

    noise_activation_and_deactivation_components = [NegativeGammaComponent(4, 5)] + deepcopy(noise_and_activation_components)
    noise_zero_activation_and_deactivation_components = [NegativeGammaComponent(4, 5)] + deepcopy(noise_zero_and_activation_components)

    pure_gaussian_mix = [GaussianComponent(0, 10), RestrictedPositiveGaussianComponent(4, 5)]

    components = noise_activation_and_deactivation_components

    em = EM(components)

    em.fit(data.ravel().squeeze())
    print em.BIC(data.ravel().squeeze())
    #print gamma.ppf(0.05, components[1].shape, scale=components[1].scale) + components[1].mu

    plt.figure()
    plt.hist(data, bins=50, normed=True)
    xRange = np.arange(math.floor(min(data)), math.ceil(max(data)), 0.1)
    pdf_sum = np.zeros(xRange.shape)
    for i, component in enumerate(em.components):
        pdf = component.pdf(xRange) * em.mix[i]
        plt.plot(xRange, pdf)
        pdf_sum += pdf
    plt.plot(xRange, pdf_sum)

    #ggm = GGGM()
    #ggm.init(random_sample)
    #ggm.estimate(random_sample)
    #
    #plt.figure()
    #plt.hist(random_sample, bins=50, normed=True)
    #pdf_sum = np.zeros(xRange.shape)
    #
    #pdf = gamma.pdf(-xRange, ggm.shape_n, scale=ggm.scale_n)*(ggm.mixt[0])
    #plt.plot(xRange, pdf)
    #pdf_sum += pdf
    #
    #pdf = norm.pdf(xRange, loc=ggm.mean, scale=math.sqrt(ggm.var))*(ggm.mixt[1])
    #plt.plot(xRange, pdf)
    #pdf_sum += pdf
    #
    #pdf = gamma.pdf(xRange, ggm.shape_p, scale=ggm.scale_p)*(ggm.mixt[2])
    #plt.plot(xRange, pdf)
    #pdf_sum += pdf
    #
    #plt.plot(xRange, pdf_sum)

    plt.show()
