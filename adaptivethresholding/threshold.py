'''
Created on 27 Apr 2010

@author: filo
'''

import numpy as np
from copy import deepcopy

from nipype.interfaces.base import BaseInterface, TraitedSpec, File, Bunch, \
    traits, OutputMultiPath
#from scikits.learn import mixture

from scipy.stats.distributions import gamma
import os
from gamma_fit import GaussianComponent, GammaComponent, \
    NegativeGammaComponent, FixedMeanGaussianComponent
from gamma_fit import EM as myEM
from nipype.interfaces.base import isdefined
from nipype.utils.filemanip import split_filename
import sys
#mpl.use("Cairo")
import pylab as plt
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import nibabel as nifti

import nipype.interfaces.utility as util     # utility
import nipype.interfaces.spm as spm     # utility
import nipype.pipeline.engine as pe          # pypeline engine

#from scipy.stats import distributions
#from scipy.optimize import brentq

import math


class ThresholdGGMMInputSpec(TraitedSpec):
    stat_image = File(exists=True, desc='stat images from a t-contrast', copyfile=True, mandatory=True)
    no_deactivation_class = traits.Bool(False, usedefault=True)
    mask_file = File(exists=True)
    models = traits.List()
    FNR_threshold = traits.Range(low=0.0, high=1.0, exclude_low=True, exclude_high=True)


class ThresholdGGMMOutputSpec(TraitedSpec):
    threshold = traits.Float()
    corrected_threshold = traits.Float()
    gaussian_mean = traits.Float()
    unthresholded_corrected_map = File(exists=True)
    thresholded_map = File(exists=True)
    histogram = OutputMultiPath(File(exists=True))
    selected_model = traits.Str()


class ThresholdGGMM(BaseInterface):
    input_spec = ThresholdGGMMInputSpec
    output_spec = ThresholdGGMMOutputSpec

    def _gen_thresholded_map_filename(self):
        _, fname, ext = split_filename(self.inputs.stat_image)
        return os.path.abspath(fname + "_thr" + ext)

    def _gen_corrected_map_filename(self):
        _, fname, ext = split_filename(self.inputs.stat_image)
        return os.path.abspath(fname + "_corr" + ext)

    def _fit_model(self, masked_data, components, label):
        em = myEM(components)
        em.fit(masked_data)
        gamma_gauss_pp = em.posteriors(masked_data)
        bic = em.BIC(masked_data)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.hist(masked_data, bins=50, normed=True)
        xRange = np.arange(math.floor(min(masked_data)), math.ceil(max(masked_data)), 0.1)
        pdf_sum = np.zeros(xRange.shape)
        for i, component in enumerate(em.components):
            pdf = component.pdf(xRange) * em.mix[i]
            plt.plot(xRange, pdf)
            pdf_sum += pdf

        at = AnchoredText("BIC = %f" % (bic),
                  loc=1, prop=dict(size=8), frameon=True,
                  )
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)
        plt.plot(xRange, pdf_sum)
        plt.xlabel("T values")
        plt.savefig("histogram%s.pdf" % label)
        return gamma_gauss_pp, bic, ax

    def _run_interface(self, runtime):
        if isdefined(self.inputs.mask_file):
            img = nifti.load(self.inputs.mask_file)
            mask = np.array(img.get_data()) == 1
        else:
            print "self.inputs.stat_imagese = %s" % str(self.inputs.stat_image)
            img = nifti.load(self.inputs.stat_image)
            mask = np.ones(img.get_data().shape) == 1

        fname = self.inputs.stat_image
        img = nifti.load(fname)
        data = np.array(img.get_data())

        masked_data = data[mask > 0].ravel().squeeze()

        no_signal_components = [GaussianComponent(0, 10)]
        no_signal_zero_components = [FixedMeanGaussianComponent(0, 10)]

        noise_and_activation_components = deepcopy(no_signal_components)
        noise_and_activation_components.append(GammaComponent(4, 5))
        noise_zero_and_activation_components = deepcopy(no_signal_zero_components)
        noise_zero_and_activation_components.append(GammaComponent(4, 5))

        noise_activation_and_deactivation_components = [NegativeGammaComponent(4, 5)] + deepcopy(noise_and_activation_components)
        noise_zero_activation_and_deactivation_components = [NegativeGammaComponent(4, 5)] + deepcopy(noise_zero_and_activation_components)

        best = (None, sys.maxint, None)

        for model in self.inputs.models:
            components = {'no_signal': no_signal_components,
                          'no_signal_zero': no_signal_zero_components,
                          'noise_and_activation': noise_and_activation_components,
                          'noise_zero_and_activation': noise_zero_and_activation_components,
                          'noise_activation_and_deactivation': noise_activation_and_deactivation_components,
                          'noise_zero_activation_and_deactivation': noise_zero_activation_and_deactivation_components}
            gamma_gauss_pp, bic, ax = self._fit_model(masked_data, components[model], label=model)
            if bic < best[1]:
                best = (gamma_gauss_pp, bic, model)

        gamma_gauss_pp = best[0]
        self._selected_model = best[2]
        thresholded_map = np.zeros(data.shape)
        if isdefined(self.inputs.FNR_threshold):
            if self._selected_model.endswith('activation_and_deactivation'):
                self._gaussian_mean = components[self._selected_model][1].mu
                self._threshold = gamma.ppf(self.inputs.FNR_threshold, components[2].shape, scale=components[2].scale) + components[2].mu
                thresholded_map[data > self._threshold] = data[data > self._threshold]
            elif self._selected_model.endswith('and_activation'):
                self._gaussian_mean = components[self._selected_model][0].mu
                self._threshold = gamma.ppf(self.inputs.FNR_threshold, components[1].shape, scale=components[1].scale) + components[1].mu
                thresholded_map[data > self._threshold] = data[data > self._threshold]
            else:
                self._gaussian_mean = components[self._selected_model][0].mu
                self._threshold = masked_data.max() + 1
        else:
            active_map = np.zeros(data.shape) == 1
            if self._selected_model.endswith('activation_and_deactivation'):
                active_map[mask] = np.logical_and(gamma_gauss_pp[:, 2] > gamma_gauss_pp[:, 1], np.logical_and(gamma_gauss_pp[:, 2] > gamma_gauss_pp[:, 0], data[mask] > 0.01))
                self._gaussian_mean = components[self._selected_model][1].mu
            elif self._selected_model.endswith('and_activation'):
                active_map[mask] = np.logical_and(gamma_gauss_pp[:, 1] > gamma_gauss_pp[:, 0], data[mask] > 0.01)
                self._gaussian_mean = components[self._selected_model][0].mu
            else:
                self._gaussian_mean = components[self._selected_model][0].mu

            thresholded_map[active_map] = data[active_map]
            if active_map.sum() != 0:
                self._threshold = data[active_map].min()
            else:
                self._threshold = masked_data.max() + 1 #setting artificially high threshold

        self._corrected_threshold = self._threshold + self._gaussian_mean
        #output = open(fname+'threshold.pkl', 'wb')
        #cPickle.dump(self._threshold, output)
        #output.close()

        #plt.axvline(self._threshold, color='r')


#        at = AnchoredText("BIC = %f, th = %f"%(bic,self._threshold),
#                  loc=1, prop=dict(size=8), frameon=True,
#                  )
#        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
#        ax.add_artist(at)
#        plt.savefig("histogram.pdf")
        thresholded_map = np.reshape(thresholded_map, data.shape)

        new_img = nifti.Nifti1Image(thresholded_map, img.get_affine(), img.get_header())
        nifti.save(new_img, self._gen_thresholded_map_filename())

        corrected_data = data.copy()
        corrected_data[mask > 0] += self._gaussian_mean
        new_img = nifti.Nifti1Image(corrected_data, img.get_affine(), img.get_header())
        nifti.save(new_img, self._gen_corrected_map_filename())

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        fname = self.inputs.stat_image
        outputs['threshold'] = float(self._threshold)
        outputs['corrected_threshold'] = float(self._corrected_threshold)
        outputs['gaussian_mean'] = float(self._gaussian_mean)
        outputs['thresholded_map'] = self._gen_thresholded_map_filename()
        outputs['unthresholded_corrected_map'] = self._gen_corrected_map_filename()
        outputs['selected_model'] = self._selected_model
        outputs['histogram'] = []
        for model in self.inputs.models:
            outputs['histogram'].append(os.path.realpath('histogram%s.pdf' % model))
        return outputs


def CreateTopoFDRwithGGMM(name="topo_fdr_with_ggmm", correct=False):

    inputnode = pe.Node(interface=util.IdentityInterface(fields=['stat_image', "spm_mat_file", "contrast_index", "mask_file"]), name="inputnode")

    ggmm = pe.MapNode(interface=ThresholdGGMM(no_deactivation_class=False,
                                              models=['noise_and_activation',
                                                        'noise_activation_and_deactivation',
                                                        'no_signal', ]), name="ggmm", iterfield=['stat_image'])


    topo_fdr = pe.MapNode(interface=spm.Threshold(), name="topo_fdr", iterfield=['stat_image', 'contrast_index', 'height_threshold'])
    topo_fdr.inputs.use_fwe_correction = False
    topo_fdr.inputs.force_activation = True
    topo_fdr.inputs.height_threshold_type = 'stat'


    topo_fdr_with_ggmm = pe.Workflow(name=name)

    topo_fdr_with_ggmm.connect([(inputnode, ggmm, [('stat_image', 'stat_image'),
                                                    ('mask_file', 'mask_file')
                                                    ]),

                           (inputnode, topo_fdr, [('spm_mat_file', 'spm_mat_file'),
                                                  ('contrast_index', 'contrast_index')])
                           ])
    if correct:
        topo_fdr_with_ggmm.connect([
                           (ggmm, topo_fdr, [('corrected_threshold', 'height_threshold'),
                                            ('unthresholded_corrected_map', 'stat_image')])
                           ])
    else:
        topo_fdr_with_ggmm.connect([
                            (inputnode, topo_fdr, [('stat_image', 'stat_image')
                                                    ]),
                           (ggmm, topo_fdr, [('threshold', 'height_threshold')])
                           ])

    return topo_fdr_with_ggmm
