#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
from astropy.cosmology import Planck15 as cosmo

"""
The code will be divided between
Fitter & Model

The Fitter loads a model that it could fit.

Info:
- Modefit.BaseFitter is minimizing
`-2*self.model.get_logprob(*self._get_model_args_)`
"""

from modefit.baseobjects import BaseModel, BaseFitter, DataHandler, BaseObject


"""
USAGE:

model = baserate.BaseRateModel()
ratefitter = baserate.RateFitter()
ratefitter.set_data(ncounts, redshift_ranges)
ratefitter.set_model(model)


ratefitter.set_fitted_flag(ratefitter._central_redshiftranges<0.7)
ratefitter.fit(a_guess=0.2)

_ = ratefitter.show()
"""


#
# FITTER
#

class RateFitter(BaseFitter):
    """ """
    PROPERTIES = ["counts", "redshift_ranges"]
    SIDE_PROPERTIES = ["fitted_flag"]
    DERIVED_PROPERTIES = ["fitted_counts", "fitted_redshift_ranges"]

    # ================ #
    #  BaseModel Struc #
    # ================ #

    def _get_model_args_(self):
        """ """
        return self.fitted_counts, self.fitted_redshift_ranges

    # ================ #
    #  Methods         #
    # ================ #

    # --------- #
    #  SETTER   #
    # --------- #

    def set_data(self, counts, redshift_ranges):
        """ Number of counts in your sample within the given redshift_ranges"""
        self._properties["counts"] = counts
        self._properties["redshift_ranges"] = np.asarray(redshift_ranges)

    def set_fitted_flag(self, fitted_flag):
        """ """
        if len(fitted_flag) != self.ndata:
            raise ValueError(\
       "Size of the data (%d) do not match that of the given fitted_flag (%d)"\
            %(self.ndata,len(fitted_flag)))

        self._side_properties["fitted_flag"] =\
                              fitted_flag
        self._derived_properties["fitted_counts"] =\
                                 self.counts[self.fitted_flag]
        self._derived_properties["fitted_redshift_ranges"] =\
                                 self.redshift_ranges.T[self.fitted_flag].T

    # --------- #
    #  PLOTTER  #
    # --------- #

    def show(self, datacolor="C0", modelcolor="C1", stepalpha=0.2,\
             add_proba=True):
        """ """
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=[10,6])

        ax = fig.add_axes([0.12, 0.15, 1-0.12*2, 0.75])
        ax.step(self._central_redshiftranges, self.counts, where="mid",\
                alpha=stepalpha)

        if self.fitted_flag is not None and not np.all(self.fitted_flag):
            prop = dict(marker="o", facecolors="None", edgecolors=datacolor)

            ax.scatter(self._central_redshiftranges[~self.fitted_flag],\
                       self.counts[~self.fitted_flag], **prop)

            prop["edgecolors"] = modelcolor

            ax.scatter(self._central_redshiftranges[~self.fitted_flag],\
                       self.get_model(self.redshift_ranges.T\
                                      [~self.fitted_flag].T),**prop)

        prop = dict(marker="o")

        ax.scatter(self._central_redshiftranges[self.fitted_flag],\
                   self.fitted_counts, c=datacolor, **prop)
        ax.scatter(self._central_redshiftranges[self.fitted_flag],\
                   self.get_model(self.fitted_redshift_ranges),c=modelcolor,\
                   **prop)

        ax.set_ylim(0)
        ax.set_ylabel("counts", fontsize = 20)
        ax.set_xlabel("redshift", fontsize = 20)

        ax.tick_params(axis = 'both',
                       direction = 'in',
                       length = 10, width = 3,
                       labelsize = 20,
                       which = 'both',
                       top = True, right = True)

        if add_proba:
            axt = ax.twinx()
            if self.fitted_flag is not None and not np.all(self.fitted_flag):

                axt.plot(self._central_redshiftranges[self.fitted_flag],
                         self.model.get_cumuprob(self.fitted_counts,\
                                                 self.fitted_redshift_ranges),
                         color="0.5",ls="-")
                axt.plot(self._central_redshiftranges,\
                         self.model.get_cumuprob(self.counts,\
                                                 self.redshift_ranges),
                         color="0.5",ls="--")
            else:
                axt.plot(self._central_redshiftranges,
                         self.model.get_cumuprob(self.counts,\
                                                 self.redshift_ranges),
                         color="0.5",ls="-")

            axt.set_ylabel("Poisson cdf", fontsize = 20)
            axt.tick_params(labelsize = 20)

        return {"fig":fig, "ax":ax}

    def pshow(self, guess):
        """ """
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize = [10,6])

        num_plots = len(self.redshift_ranges.T)

        colormap = plt.cm.gist_ncar
        plt.gca().set_prop_cycle(plt.cycler('color',\
                                            plt.cm.jet(np.linspace(0, 1, num_plots))))

        labels = []

        for i in range(1,num_plots):
            self.set_fitted_flag(self._central_redshiftranges\
                               < self._central_redshiftranges[i])

            labels.append('fit on ' + str(i) + ' bins')

            self.fit(a_guess = guess)
            plt.plot(self._central_redshiftranges[self.fitted_flag],\
                     self.model.get_cumuprob(self.fitted_counts,\
                                             self.fitted_redshift_ranges))

        ax = plt.gca()
        ax.tick_params(axis = 'both',
                       direction = 'in',
                       length = 10, width = 3,
                       labelsize = 20,
                       which = 'both',
                       top = True, right = True)
        plt.xlabel('$z_{max}$', fontsize = 20)
        plt.ylabel('Poisson cdf', fontsize = 20)

        plt.title('Evolution of poisson cdf with bins used to fit', fontsize = 20)

        plt.legend(labels, ncol=1, loc='upper right', 
                   #bbox_to_anchor=[0.5, 1.1], 
                   columnspacing=1.0, labelspacing=0.0,
                   handletextpad=0.0, handlelength=1.5,
                   fancybox=True, shadow=True)

        plt.show()

    # ================ #
    #  Parameters      #
    # ================ #

    # - Data In
    @property
    def counts(self):
        """ Number of observation within a given redshift_ranges """
        return self._properties["counts"]

    @property
    def redshift_ranges(self):
        """ redshift_ranges """
        return self._properties["redshift_ranges"]

    @property
    def ndata(self):
        """ len(self.counts) """
        return len(self.counts)

    @property
    def _central_redshiftranges(self):
        """ """
        return np.mean(self.redshift_ranges, axis=0)

    # - Fitted parameters
    @property
    def fitted_flag(self):
        """ """
        return self._side_properties["fitted_flag"]

    @property
    def fitted_counts(self):
        """ """
        if self._derived_properties["fitted_counts"] is None:
            self._derived_properties["fitted_counts"] = self.counts
        return self._derived_properties["fitted_counts"]

    @property
    def fitted_redshift_ranges(self):
        """ """
        if self._derived_properties["fitted_redshift_ranges"] is None:
            self._derived_properties["fitted_redshift_ranges"] =\
            self.redshift_ranges
        return self._derived_properties["fitted_redshift_ranges"]

#
# MODEL
#

class _RateModelStructure_( BaseModel ):
    """ """
    VOLUME_SCALE     = 1e8
    RATEPARAMETERS   = []
    MISSEDPARAMETERS = []

    def __new__(cls,*arg,**kwarg):
        """ Black Magic allowing generalization of Polynomial models """
        cls.FREEPARAMETERS = cls.RATEPARAMETERS + cls.MISSEDPARAMETERS
        return super(_RateModelStructure_,cls).__new__(cls)

    def setup(self, parameters):
        """ """
        if self.nmissedparameters>0:
            self.paramrate   = {k:v for k,v in\
                                zip(self.RATEPARAMETERS,\
                                parameters[:self.nrateparameters])}
            self.parammissed = {k:v for k,v in\
                                zip(self.MISSEDPARAMETERS,\
                                parameters[self.nrateparameters:])}

        elif self.nrateparameters==1:
            self.paramrate   = {self.RATEPARAMETERS[0]: parameters}

        else:
            self.paramrate   = {k:v for k,v in zip(self.RATEPARAMETERS,\
                                                   parameters)}

    # --------- #
    #  Model    #
    # --------- #

    def get_model(self, *args):
        """ """
        return self.get_expectedrate(*args)

    #  Nature - Missed
    def get_expectedrate(self, redshift_ranges):
        """ """
        return self.get_rate( redshift_ranges )\
             - self.get_missedrate( redshift_ranges )

    def get_rate(self, redshift_ranges):
        """ """
        # Could use self.paramrate
        raise\
        NotImplementedError("Your RateModel class must define `get_rate()` ")

    def get_missedrate(self, redshift_ranges):
        """ """
        # Could use self.missedrate
        raise\
    NotImplementedError("Your RateModel class must define `get_missedrate()` ")

    # ----------- #
    # Likelihood  #
    # ----------- #

    def get_logprob(self, *args ):
        """ """
        return self.get_loglikelihood(*args) + self.get_logpriors()

    def get_logpriors(self):
        """ """
        return 0

    def get_loglikelihood(self, counts, redshift_ranges ):
        """ """
        # poisson.pmf(k, mu) = probability to observe k counts\
                             # given that we expect mu
        return np.sum(np.log(self.get_probabilities(counts,redshift_ranges)))

    def get_probabilities(self, counts, redshift_ranges):
        """ Returns the poisson statistics applied to each cases """
        return stats.poisson.pmf(counts,self.get_expectedrate(redshift_ranges))

    def get_cumuprob(self, counts, redshift_ranges):
        """ Returns the cumulative poisson statistics """
        return stats.poisson.cdf(counts,self.get_expectedrate(redshift_ranges))

    # ================= #
    #  Properties       #
    # ================= #

    @property
    def nrateparameters(self):
        """ """
        return len(self.RATEPARAMETERS)

    @property
    def nmissedparameters(self):
        """ """
        return len(self.MISSEDPARAMETERS)

# ==================== #
#                      #
#  RATE MODELS         #
#                      #
# ==================== #

class VolumeRateModel( BaseObject ):
    """ """
    RATEPARAMETERS = ["a"]

    # boundaries
    a_boundaries = [0,None]

    def get_logprior(self):
        """ """
        return 0

    def _get_rate_(self, redshifts):
        """ """
        return self.paramrate['a']*\
               cosmo.comoving_volume(redshifts).value/self.VOLUME_SCALE

    def get_rate(self, redshift_ranges):
        """ """
        # Could use self.paramrate
        return self._get_rate_(redshift_ranges[1])\
             - self._get_rate_(redshift_ranges[0])


class PerrettRateModel( VolumeRateModel ):
    """ """
    RATEPARAMETERS = ["a"]
    def _get_rate_(self, redshifts):
        """ See eq. 6 of https://arxiv.org/pdf/1811.02379.pdf """
        return self.paramrate['a']* (1.75 *1e-5* (1+redshifts)**2.11 )

# ==================== #
#                      #
#  MISSED MODELS       #
#                      #
# ==================== #

class NoMissedModel( BaseObject ):
    """ """
    MISSEDPARAMETERS = []

    def get_logprior(self):
        """ """
        return 0

#    def _get_missedrate_(self, redshifts):
#        """ """
#        return 0

    def get_missedrate(self, redshift_ranges):
        """ """
        return 0

class ConstMissedModel( BaseObject ):
    """ """
    MISSEDPARAMETERS = ['zmax']

    def get_logprior(self):
        """ """
        return 0

    def _get_missedrate_(self, redshifts):
        """ """
        flag_up = redshifts > self.parammissed['zmax']
        missed = np.zeros(len(redshifts))

        missed[flag_up] = 10

        return missed

    def get_missedrate(self, redshift_ranges):
        """ """
        # Could use self.paramrate
        return self._get_missedrate_(redshift_ranges[1])\

class ExpoMissedModel( NoMissedModel ):
    """ """
    MISSEDPARAMETERS = ["b", "zmax", "zc"]

    #boundaries
    b_boundaries = [0,None]
    zmax_boundaries = [0,None]
    zc_boundaries = [1e-5,None]

    def get_logprior(self):
        """ """
        return 0

    def _get_missedrate_(self, redshifts):
        """ """
        flag_up = redshifts > self.parammissed['zmax']
        missed = np.zeros(len(redshifts))

        missed[flag_up] = \
        self.parammissed['b']/self.VOLUME_SCALE\
      * np.exp(redshifts[flag_up]/self.parammissed['zc'])

        return missed

    def get_missedrate(self, redshift_ranges):
        """ """
        return self._get_missedrate_(redshift_ranges[1])\
#             - self._get_missedrate_(redshift_ranges[0])

# ==================== #
#                      #
#   TOTAL MODELS       #
#                      #
# ==================== #

class BaseRateModel(NoMissedModel,VolumeRateModel,_RateModelStructure_):
    """ """
class PerrettRateModel(NoMissedModel,PerrettRateModel,_RateModelStructure_):
    """ """
class ExpoRateModel(ExpoMissedModel,VolumeRateModel,_RateModelStructure_):
    """ """
class ConstRateModel(ConstMissedModel,VolumeRateModel,_RateModelStructure_):
    """ """
