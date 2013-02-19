from adaptivethresholding.gamma_fit import GaussianComponent,\
    FixedMeanGaussianComponent, GammaComponent, NegativeGammaComponent,\
    RestrictedPositiveGaussianComponent, EM
from copy import deepcopy
import pylab as plt


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

components = noise_and_activation_components

em = EM(components)

import numpy as np
import pylab as plt
signal = np.random.normal(loc=3, size=10000**2)
noise = np.random.normal(size=900000)

data = np.hstack((signal,noise))

p = plt.hist(data, bins=50)

plt.matshow(data.reshape(100,-1))

em.fit(data.ravel().squeeze())

plt.matshow(em.mix[:,0].reshape(100,100))
plt.matshow(em.mix[:,1].reshape(100,100))

posteriors = em.posteriors(data.ravel().squeeze())

plt.matshow(posteriors[:,0].reshape(100,100))
plt.matshow(posteriors[:,1].reshape(100,100))
plt.matshow((posteriors[:,0] > posteriors[:,1]).reshape(100,100))

em = EM(components, mix_type="single")
em.fit(data.ravel().squeeze())

posteriors = em.posteriors(data.ravel().squeeze())

plt.matshow(posteriors[:,0].reshape(100,100))
plt.matshow(posteriors[:,1].reshape(100,100))

plt.matshow((posteriors[:,0] > posteriors[:,1]).reshape(100,100))

plt.show()