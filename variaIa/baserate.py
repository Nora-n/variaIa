#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
from astropy.cosmology import Planck15 as cosmo
from modefit.baseobjects import BaseModel, BaseFitter, BaseObject
import matplotlib.pyplot as plt


"""
The code will be divided between
Fitter & Model

The Fitter loads a model that it could fit.

Info:
- Modefit.BaseFitter is minimizing
`-2*self.model.get_logprob(*self._get_model_args_)`
"""

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


# =========================================================================== #
#                                                                             #
#                                   FITTER                                    #
#                                                                             #
# =========================================================================== #

class RateFitter(BaseFitter):
    """ """
    PROPERTIES = ["counts", "redshift_ranges"]
    SIDE_PROPERTIES = ["fitted_flag"]
    DERIVED_PROPERTIES = ["fitted_counts", "fitted_redshift_ranges"]

    # =================================================================== #
    #                           BaseModel Struc                           #
    # =================================================================== #

    def _get_model_args_(self):
        """ """
        return self.fitted_counts, self.fitted_redshift_ranges

    # =================================================================== #
    #                               Methods                               #
    # =================================================================== #

    # ------------------------------------------------------------------- #
    #                               SETTER                                #
    # ------------------------------------------------------------------- #

    def set_data(self, counts, redshift_ranges):
        """ Number of counts in your sample within the given redshift_ranges"""
        self._properties["counts"] = counts
        self._properties["redshift_ranges"] = np.asarray(redshift_ranges)

    def set_fitted_flag(self, fitted_flag):
        """ """
        if len(fitted_flag) != self.ndata:
            raise ValueError(
                "Data size (%d) doen't match the given fitted_flag (%d)"
                % (self.ndata, len(fitted_flag)))

        self._side_properties["fitted_flag"] =\
            fitted_flag
        self._derived_properties["fitted_counts"] =\
            self.counts[self.fitted_flag]
        self._derived_properties["fitted_redshift_ranges"] =\
            self.redshift_ranges.T[self.fitted_flag].T

    # ------------------------------------------------------------------- #
    #                               PLOTTER                               #
    # ------------------------------------------------------------------- #

    def show(self, ax=None,
             datacolor="C0", modelcolor="C1", stepalpha=0.2,
             add_proba=True):
        """ """
        if ax is None:
            fig = plt.figure(figsize=[7, 3.5])
            ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])

        ax.step(self._central_redshiftranges, self.counts, where="mid",
                color=datacolor, alpha=stepalpha)

        if self.fitted_flag is not None and not np.all(self.fitted_flag):
            prop = dict(marker="o", facecolors="None", edgecolors=datacolor)

            ax.scatter(self._central_redshiftranges[~self.fitted_flag],
                       self.counts[~self.fitted_flag], **prop)

            prop["edgecolors"] = modelcolor

            ax.scatter(self._central_redshiftranges[~self.fitted_flag],
                       self.get_model(self.redshift_ranges.T
                                      [~self.fitted_flag].T), **prop)

        prop = dict(marker="o")

        ax.scatter(self._central_redshiftranges[self.fitted_flag],
                   self.fitted_counts,
                   color=datacolor, **prop)
        ax.scatter(self._central_redshiftranges[self.fitted_flag],
                   self.get_model(self.fitted_redshift_ranges),
                   color=modelcolor, **prop)

        ax.set_ylim(0)
        ax.set_ylabel(r'$\mathrm{counts}$', fontsize='x-large')
        ax.set_xlabel(r'$\mathrm{redshift}$', fontsize='x-large')

        ax.tick_params(labelsize='x-large')

        if add_proba:
            axt = ax.twinx()
            if self.fitted_flag is not None and not np.all(self.fitted_flag):

                axt.plot(self._central_redshiftranges[self.fitted_flag],
                         self.model.get_cumuprob(self.fitted_counts,
                                                 self.fitted_redshift_ranges),
                         color="0.5", ls="-")
                axt.plot(self._central_redshiftranges,
                         self.model.get_cumuprob(self.counts,
                                                 self.redshift_ranges),
                         color="0.5", ls="--")
            else:
                axt.plot(self._central_redshiftranges,
                         self.model.get_cumuprob(self.counts,
                                                 self.redshift_ranges),
                         color="0.5", ls="-")

            axt.set_ylabel(r'$\mathrm{Poisson\,\,cdf}$', fontsize='x-large')
            axt.tick_params(labelsize=15)

    def pshow(self, guess):
        """ """
        fig = plt.figure(figsize=[10, 6])
        ax = fig.add_axes([.1, .12, .8, .8])

        num_plots = len(self.redshift_ranges.T)

        plt.gca().set_prop_cycle(plt.cycler('color',
                                            plt.cm.jet(np.linspace(0, 1,
                                                       num_plots))))

        labels = []

        for i in range(1, num_plots):
            self.set_fitted_flag(self._central_redshiftranges
                                 < self._central_redshiftranges[i])

            labels.append('fit on ' + str(i) + ' bins')

            self.fit(a_guess=guess)
            plt.plot(self._central_redshiftranges[self.fitted_flag],
                     self.model.get_cumuprob(self.fitted_counts,
                                             self.fitted_redshift_ranges))

        ax.tick_params(direction='in',
                       length=5, width=1,
                       labelsize=12,
                       top=True, right=True)

        ax.set_xlabel(r'$z_{max}$', fontsize='x-large')
        ax.set_ylabel(r'$\mathrm{Poisson\,\,cdf}$', fontsize='x-large')

        plt.title(r'$\mathrm{Evolution\,\,of\,\,poisson\,\,cdf\,\,with\,\,}$' +
                  r'$\mathrm{bins\,\,used\,\,to\,\,fit}$', fontsize='x-large')

        plt.legend(labels, ncol=1, loc='upper right',
                   columnspacing=1.0, labelspacing=0.0,
                   handletextpad=0.0, handlelength=1.5,
                   fancybox=True, shadow=True)

    # =================================================================== #
    #                               Parameters                            #
    # =================================================================== #

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

# =========================================================================== #
#                                                                             #
#                                MODEL STRUC                                  #
#                                                                             #
# =========================================================================== #


class _RateModelStructure_(BaseModel):
    """ """
    VOLUME_SCALE = 1e8
    RATEPARAMETERS = []
    MISSEDPARAMETERS = []

    def __new__(cls, *arg, **kwarg):
        """ Black Magic allowing generalization of Polynomial models """
        cls.FREEPARAMETERS = cls.RATEPARAMETERS + cls.MISSEDPARAMETERS
        return super(_RateModelStructure_, cls).__new__(cls)

    def setup(self, parameters):
        """ """
        if self.nmissedparameters > 0:
            self.paramrate = {k: v for k, v in
                              zip(self.RATEPARAMETERS,
                                  parameters[:self.nrateparameters])}
            self.parammissed = {k: v for k, v in
                                zip(self.MISSEDPARAMETERS,
                                    parameters[self.nrateparameters:])}

        elif self.nrateparameters == 1:
            self.paramrate = {self.RATEPARAMETERS[0]: parameters}

        else:
            self.paramrate = {k: v for k, v in zip(self.RATEPARAMETERS,
                                                   parameters)}

    # ------------------------------------------------------------------- #
    #                               Model                                 #
    # ------------------------------------------------------------------- #

    def get_model(self, *args):
        """ """
        return self.get_expectedrate(*args)

    #  Nature - Missed
    def get_expectedrate(self, redshift_ranges):
        """ """
        return self.get_rate(redshift_ranges)\
            - self.get_missedrate(redshift_ranges)

    def get_rate(self, redshift_ranges):
        """ """
        # Could use self.paramrate
        raise\
            NotImplementedError(
                "Your RateModel class must define `get_rate()` ")

    def get_missedrate(self, redshift_ranges):
        """ """
        # Could use self.missedrate
        raise\
            NotImplementedError(
                "Your RateModel class must define `get_missedrate()` ")

    # ------------------------------------------------------------------- #
    #                               Likelihood                            #
    # ------------------------------------------------------------------- #

    def get_logprob(self, *args):
        """ """
        return self.get_loglikelihood(*args) + self.get_logpriors()

    def get_logpriors(self):
        """ """
        return 0

    def get_loglikelihood(self, counts, redshift_ranges):
        """ """
        # poisson.pmf(k, mu) = probability to observe k counts\
        # given that we expect mu
        return np.sum(np.log(self.get_probabilities(counts, redshift_ranges)))

    def get_probabilities(self, counts, redshift_ranges):
        """ Returns the poisson statistics applied to each cases """
        return stats.poisson.pmf(counts,
                                 self.get_expectedrate(redshift_ranges))

    def get_cumuprob(self, counts, redshift_ranges):
        """ Returns the cumulative poisson statistics """
        return stats.poisson.cdf(counts,
                                 self.get_expectedrate(redshift_ranges))

    # =================================================================== #
    #                               Properties                            #
    # =================================================================== #

    @property
    def nrateparameters(self):
        """ """
        return len(self.RATEPARAMETERS)

    @property
    def nmissedparameters(self):
        """ """
        return len(self.MISSEDPARAMETERS)


# =========================================================================== #
#                                                                             #
#                                RATE MODELS                                  #
#                                                                             #
# =========================================================================== #

class VolumeRateModel(BaseObject):
    """ """
    RATEPARAMETERS = ["a"]

    # boundaries
    a_boundaries = [0, None]

    def get_logprior(self):
        """ """
        return 0

    def _get_rate_(self, redshifts):
        """ """
        return self.paramrate['a']/self.VOLUME_SCALE *\
            cosmo.comoving_volume(redshifts).value

    def get_rate(self, redshift_ranges):
        """ """
        # Could use self.paramrate
        return self._get_rate_(redshift_ranges[1])\
            - self._get_rate_(redshift_ranges[0])


class PerrettRateModel(VolumeRateModel):
    """ """
    RATEPARAMETERS = ["a"]

    def _get_rate_(self, redshifts):
        """ See eq. 6 of https://arxiv.org/pdf/1811.02379.pdf """
        return self.paramrate['a'] * (1.75 * 1e-5 * (1+redshifts)**2.11)


# =========================================================================== #
#                                                                             #
#                                MISSED MODELS                                #
#                                                                             #
# =========================================================================== #


class NoMissedModel(BaseObject):
    """ """
    MISSEDPARAMETERS = []

    def _get_missedrate_(self, redshifts):
        """ """
        return 0

    def get_missedrate(self, redshift_ranges):
        """ """
        return 0


class ConstMissedModel(BaseObject):
    """ """
    MISSEDPARAMETERS = ['zmax']

    def _get_missedrate_(self, redshifts):
        """ """
        flag_up = redshifts > self.parammissed['zmax']
        missed = np.zeros(len(redshifts))

        missed[flag_up] = -10

        return missed

    def get_missedrate(self, redshift_ranges):
        """ """
        # Could use self.paramrate
        return self._get_missedrate_(redshift_ranges[1])


class ExpoMissedModel(BaseObject):
    """ """
    MISSEDPARAMETERS = ["b", "zmax", "zc"]

    # boundaries
    b_boundaries = [0, None]
    zmax_boundaries = [0, None]
    zc_boundaries = [1e-5, None]

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
        return self._get_missedrate_(redshift_ranges[1])


class VolumeMissedModel(BaseObject):
    """ """
    MISSEDPARAMETERS = ['b', 'zmax']

    # boundaries
    b_boundaries = [0, None]
    zmax_boundaries = [0, None]

    def _get_missedrate_(self, redshifts):
        """ """
        flag_up = redshifts > self.parammissed['zmax']
        missed = np.zeros(len(redshifts))

        missed[flag_up] = self.parammissed['b']/self.VOLUME_SCALE *\
            cosmo.comoving_volume(redshifts[flag_up]).value

        return np.asarray(missed)

    def get_missedrate(self, redshift_ranges):
        """ """
        return self._get_missedrate_(redshift_ranges[1])\
            - self._get_missedrate_(redshift_ranges[0])


# =========================================================================== #
#                                                                             #
#                                TOTAL MODELS                                 #
#                                                                             #
# =========================================================================== #


class VolumeNoModel(VolumeRateModel,
                    NoMissedModel,
                    _RateModelStructure_):
    """ """


class PerrettNoModel(PerrettRateModel,
                     NoMissedModel,
                     _RateModelStructure_):
    """ """


class VolumeExpoModel(VolumeRateModel,
                      ExpoMissedModel,
                      _RateModelStructure_):
    """ """


class VolumeConstModel(VolumeRateModel,
                       ConstMissedModel,
                       _RateModelStructure_):
    """ """


class VolumeVolumeModel(VolumeRateModel,
                        VolumeMissedModel,
                        _RateModelStructure_):
    """ """

# =========================================================================== #
#                                                                             #
#                                SHOW P-VALUE                                 #
#                                                                             #
# =========================================================================== #

    # ------------------------------------------------------------------- #
    #                               SETTER                                #
    # ------------------------------------------------------------------- #


def zmax_poisson(survey, rawdata, guess, loops, itsc):
    """Calculates the cdf evolution, and get the intersection. itsc must be a
    dict of lists of itsc_inf, itsc_med and itsc_sup.
    Returns list of z_linspace, cdf med and std, and (zinf, max, sup)"""
    import math
    from numpy.random import randint as rdint
    from scipy import interpolate

    # set empty base model and empty ratefitter to be filled
    base_r = VolumeNoModel()
    rate_r = RateFitter()

    # set empty list of lists to be filled
    x = [[] for i in range(loops)]
    y = [[] for i in range(loops)]
    p_zintp = [[] for i in range(loops)]

    if survey == 'SNLS':
        for i in range(loops):
            # define start of histogram randomly between 0.06 and 0.12
            data_r_a = rdint(math.floor(rawdata[0]*100)/2,
                             math.floor(rawdata[0]*100))/100
            # define end of histogram randomly between 1.10 and 1.15
            data_r_b = rdint(math.ceil(rawdata[-1]*10)*10,
                             (math.ceil(rawdata[-1]*10)
                              + math.floor(rawdata[0]*10)/2)*10)/100
            # complete random data is start + normal data + end
            data_r = np.append(data_r_a, np.append(rawdata, data_r_b))
            # define total number of bins
            nb_bins_r = rdint(5, 20)
            # define number of fits for each number of total bins
            nb_fits_per_dist = 10
            # get the counts and limits of bins
            counts_r, bord_r = np.asarray(np.histogram(data_r, bins=nb_bins_r))
            # define the list of lower and upper bins for ratefit to work
            bins_r = np.asarray([[bord_r[i], bord_r[i+1]]
                                 for i in range(len(bord_r)-1)]).T

            # create the according ratefitter
            rate_r.set_data(counts_r, bins_r)
            rate_r.set_model(base_r)

            for k in range(nb_fits_per_dist):
                # for each run with total number of bins, limit the fit to a
                # random bin from 3 to max
                rate_r.set_fitted_flag(rate_r._central_redshiftranges
                                       < rate_r._central_redshiftranges
                                       [rdint(3, nb_bins_r)])
                rate_r.fit(a_guess=guess)

                # fill i loop with positions of bins, k times
                x[i].append(rate_r._central_redshiftranges)
                # fill i loop with values of cdf, k times
                y[i].append(rate_r.model.get_cumuprob(rate_r.counts,
                                                      rate_r.redshift_ranges))
                # define the z list for interpolating
                z_intp = np.linspace(x[i][k][0], x[i][k][-1], 10000)
                # fill i loop with interpolation for saving
                p_zintp[i].append(interpolate.interp1d(
                                  x[i][k], y[i][k], kind='linear')(z_intp))
    else:
        for i in range(loops):
            data_r_a = rdint(math.floor(rawdata[0]*1000)/2,
                             math.floor(rawdata[0]*1000))/1000
            data_r_b = rdint(math.ceil(rawdata[-1]*100)*10,
                             (math.ceil(rawdata[-1]*100)
                              + math.floor(rawdata[0]*100)/2)*10)/1000
            data_r = np.append(data_r_a, np.append(rawdata, data_r_b))
            nb_bins_r = rdint(5, 20)
            nb_fits_per_dist = 10
            counts_r, bord_r = np.asarray(np.histogram(data_r, bins=nb_bins_r))
            bins_r = np.asarray([[bord_r[i], bord_r[i+1]]
                                 for i in range(len(bord_r)-1)]).T

            rate_r.set_data(counts_r, bins_r)
            rate_r.set_model(base_r)

            for k in range(nb_fits_per_dist):
                rate_r.set_fitted_flag(rate_r._central_redshiftranges
                                       < rate_r._central_redshiftranges
                                       [rdint(3, nb_bins_r)])
                rate_r.fit(a_guess=guess)

                x[i].append(rate_r._central_redshiftranges)
                y[i].append(rate_r.model.get_cumuprob(rate_r.counts,
                                                      rate_r.redshift_ranges))
                z_intp = np.linspace(x[i][k][0], x[i][k][-1], 10000)
                p_zintp[i].append(interpolate.interp1d(
                                  x[i][k], y[i][k], kind='linear')(z_intp))

    # compute the median of all interpolated curves
    p_med = np.median(np.median(p_zintp, axis=1), axis=0)
    # compute the standard deviation of all interpolated curves
    p_std = np.std(np.std(p_zintp, axis=1), axis=0)

    # construct list of horizontal values for intersection
    horiz_inf = [itsc[survey][0] for i in range(len(p_med))]
    horiz_med = [itsc[survey][1] for i in range(len(p_med))]
    horiz_sup = [itsc[survey][2] for i in range(len(p_med))]
    # gives the indice of intersection with p_med
    ind_med = np.argwhere(np.diff(np.sign(p_med - horiz_med))).flatten()[-1]
    # same with p_med - p_std
    ind_inf = np.argwhere(np.diff(np.sign(p_med - horiz_inf))).flatten()[-1]
    # same with p_med + p_std
    ind_sup = np.argwhere(np.diff(np.sign(p_med - horiz_sup))).flatten()[-1]
    # This gives the list of borne inf, moyenne, borne sup
    zmax = [round(z_intp[ind_inf], 4),
            round(z_intp[ind_med], 4),
            round(z_intp[ind_sup], 4)]

    return(z_intp, p_med, p_std, zmax)

    # ------------------------------------------------------------------- #
    #                               PLOTTER                               #
    # ------------------------------------------------------------------- #


def zmax_pshow(z_lins, meds, stds, z_max, itsc,
               show_itsc=True, show_infsup=True):
    """Plot all (or one) cdf evolution"""
    fig = plt.figure(figsize=[8, 5])
    ax = fig.add_axes([0.1, 0.12, 0.8, 0.8])

    surveys = list(z_lins.keys())

    smap = plt.cm.get_cmap('cividis')
    colors = {'SDSS': smap(0.1),
              'PS1': smap(0.5),
              'SNLS': smap(0.8)}

    for survey in surveys:
        ax.plot(z_lins[survey],
                meds[survey],
                '-', color=colors[survey],
                label=survey)
        ax.fill_between(z_lins[survey],
                        meds[survey] - stds[survey],
                        meds[survey] + stds[survey],
                        color=colors[survey], alpha=0.2)
        if show_itsc:
            ax.vline(z_max[survey][1],
                     color=colors[survey],
                     lw=1.0)
            ax.plot(z_max[survey][1],
                    itsc[survey][1],
                    color="black", marker='o')
            ax.hline(itsc[survey][1],
                     color="0.3", ls='--', lw=1.0)
# label=r"$z_{\mathrm{max,med}}$")
        if show_infsup:
            ax.vline(z_max[survey][0],
                     color=colors[survey],
                     lw=1.0)
            ax.plot(z_max[survey][0],
                    itsc[survey][0],
                    color=".7", marker='o')
            ax.hline(itsc[survey][0],
                     color="0.3", ls='-.', lw=1.0)
# label=r"$z_{\mathrm{max,inf/sup}}$")
            ax.vline(z_max[survey][2],
                     color=colors[survey],
                     lw=1.0)
            ax.plot(z_max[survey][2],
                    itsc[survey][2],
                    color=".7", marker='o')
            ax.hline(itsc[survey][2],
                     color="0.3", ls='-.', lw=1.0)

    ax.tick_params(labelsize='x-large')

    ax.set_xlim(np.min(list(z_lins.values())),
                np.max(list(z_lins.values())))
    ax.set_ylim(np.min(np.asarray(list(meds.values()))
                       - np.asarray(list(stds.values()))),
                np.max(np.asarray(list(meds.values()))
                       + np.asarray(list(stds.values()))))

    ax.set_xlabel('redshift', fontsize='x-large')
    ax.set_ylabel('Poisson cdf', fontsize='x-large')

    ax.legend(ncol=1, loc='upper right')

    ax.set_title('Statistical evolution of redshift limit',
                 fontsize='x-large')
