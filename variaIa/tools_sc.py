#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd
import ipywidgets as ipw

from variaIa import stretchevol

d = pd.read_csv('../../../Data/sne/data_cheat.csv', sep=' ', index_col='CID')
d_snf = pd.read_csv('../../../Data/sne/lssfr_paper_full_sntable.csv', sep=',')

surveys = ['SNF', 'LOWZ', 'SDSS', 'PS1', 'SNLS', 'HST']
nsurveys = ['nSNF', 'nLOWZ', 'nSDSS', 'nPS1', 'nSNLS', 'nHST']

su = ipw.Dropdown(options=surveys + ['All'] + nsurveys,
                  description='Survey:',
                  value='SNF')

cons = ipw.Checkbox(value=False,
                    description='Super conservative')

raw_df_snf = d_snf.loc[d_snf['name'].str.contains(
    'SNF|LSQ|PTF', na=False, regex=True)]
surv = {'SNF':   raw_df_snf,  # [raw_df_snf['salt2.Color'] < 0.3],
        'LOWZ': d[d['IDSURVEY'].isin([5, 61, 62, 63, 64, 65, 66])],
        'SDSS':  d[d['IDSURVEY'] == 1],
        'PS1':   d[d['IDSURVEY'] == 15],
        'SNLS':  d[d['IDSURVEY'] == 4],
        'HST':   d[d['IDSURVEY'].isin([101, 100, 106])]}

with open('../../../Data/zmax/zmax_mlim', 'rb') as f:
    z_max = pickle.load(f)
z_max['HST'] = [10, 10]
z_max['LOWZ'] = [10, 10]
z_max['SDSS'][0] = 0.10
z_max['PS1'][0] = 0.20
z_max['SNLS'][0] = 0.30

zmax_cuts = dict()
z_zcuts = dict()
x1_zcuts = dict()
x1_err_zcuts = dict()
c_zcuts = dict()
c_err_zcuts = dict()
M_zcuts = dict()
M_err_zcuts = dict()

z_zcuts['SNF'] = surv['SNF']['host.zcmb'].values
x1_zcuts['SNF'] = surv['SNF']['salt2.X1'].values
x1_err_zcuts['SNF'] = surv['SNF']['salt2.X1.err'].values
c_zcuts['SNF'] = surv['SNF']['salt2.Color'].values
c_err_zcuts['SNF'] = surv['SNF']['salt2.Color.err'].values
M_zcuts['SNF'] = surv['SNF']['gmass'].values
M_err_zcuts['SNF'] = np.sqrt((surv['SNF']['gmass.err_down'].values**2 +
                              surv['SNF']['gmass.err_up'].values**2)/2)


def df_cons(cons):
    names = []
    stretchs = []
    stretchs_err = []
    colors = []
    colors_err = []
    hostmass = []
    hostmass_err = []
    redshifts = []
    infor = list(surv['SNF']['p(prompt)'])
    py = list(surv['SNF']['p(prompt)'])
    lssfr = list(surv['SNF']['lssfr'])
    lssfr_err_d = list(surv['SNF']['lssfr.err_down'])
    lssfr_err_u = list(surv['SNF']['lssfr.err_up'])

    if cons:
        for survey in surveys[1:]:
            zmax_cuts[survey] = np.where(
                surv[survey].zCMB.values < z_max[survey][0])
            z_zcuts[survey] = surv[survey].zCMB.values[zmax_cuts[survey]]
            x1_zcuts[survey] = surv[survey].x1.values[zmax_cuts[survey]]
            x1_err_zcuts[survey] = surv[survey].x1ERR.values[zmax_cuts[survey]]
            c_zcuts[survey] = surv[survey].c.values[zmax_cuts[survey]]
            c_err_zcuts[survey] = surv[survey].cERR.values[zmax_cuts[survey]]
            M_zcuts[survey] =\
                surv[survey].HOST_LOGMASS.values[zmax_cuts[survey]]
            M_err_zcuts[survey] =\
                surv[survey].HOST_LOGMASS_ERR.values[zmax_cuts[survey]]
    else:
        for survey in surveys[1:]:
            zmax_cuts[survey] = np.where(
                surv[survey].zCMB.values < z_max[survey][-1])
            z_zcuts[survey] = surv[survey].zCMB.values[zmax_cuts[survey]]
            x1_zcuts[survey] = surv[survey].x1.values[zmax_cuts[survey]]
            x1_err_zcuts[survey] = surv[survey].x1ERR.values[zmax_cuts[survey]]
            c_zcuts[survey] = surv[survey].c.values[zmax_cuts[survey]]
            c_err_zcuts[survey] = surv[survey].cERR.values[zmax_cuts[survey]]
            M_zcuts[survey] =\
                surv[survey].HOST_LOGMASS.values[zmax_cuts[survey]]
            M_err_zcuts[survey] =\
                surv[survey].HOST_LOGMASS_ERR.values[zmax_cuts[survey]]

    for survey in surveys:
        names += [survey for i in range(len(z_zcuts[survey]))]
        stretchs += list(x1_zcuts[survey])
        stretchs_err += list(x1_err_zcuts[survey])
        colors += list(c_zcuts[survey])
        colors_err += list(c_err_zcuts[survey])
        hostmass += list(M_zcuts[survey])
        hostmass_err += list(M_err_zcuts[survey])
        redshifts += list(z_zcuts[survey])
        if survey != 'SNF':
            infor += list(stretchevol.Evol2G2M2S.delta(z_zcuts[survey]))
            py += list([0 for i in range(len(z_zcuts[survey]))])
            lssfr += list([0 for i in range(len(z_zcuts[survey]))])
            lssfr_err_d += list([0 for i in range(len(z_zcuts[survey]))])
            lssfr_err_u += list([0 for i in range(len(z_zcuts[survey]))])

    df = pd.DataFrame({'survey': names,
                       'stretchs': stretchs,
                       'stretchs_err': stretchs_err,
                       'colors': colors,
                       'colors_err': colors_err,
                       'hostmass': hostmass,
                       'hostmass_err': hostmass_err,
                       'redshifts': redshifts,
                       'infor': infor,
                       'py': py,
                       'lssfr': lssfr,
                       'lssfr_err_d': lssfr_err_d,
                       'lssfr_err_u': lssfr_err_u})
    return(df)


df_nc = df_cons(False)[df_cons(False)['survey'] != 'LOWZ']
df_c = df_cons(True)[df_cons(True)['survey'] != 'LOWZ']

names = ['SNF' for i in range(len(z_zcuts['SNF']))]
stretchs = list(x1_zcuts['SNF'])
stretchs_err = list(x1_err_zcuts['SNF'])
colors = list(c_zcuts['SNF'])
colors_err = list(c_err_zcuts['SNF'])
hostmass = list(M_zcuts['SNF'])
hostmass_err = list(M_err_zcuts['SNF'])
redshifts = list(z_zcuts['SNF'])
infor = list(surv['SNF']['p(prompt)'])
py = list(surv['SNF']['p(prompt)'])
lssfr = list(surv['SNF']['lssfr'])
lssfr_err_d = list(surv['SNF']['lssfr.err_down'])
lssfr_err_u = list(surv['SNF']['lssfr.err_up'])

for survey in surveys[2:]:
    names += [survey for i in range(len(surv[survey].zCMB.values))]
    stretchs += list(surv[survey].x1.values)
    stretchs_err += list(surv[survey].x1ERR.values)
    colors += list(surv[survey].c.values)
    colors_err += list(surv[survey].cERR.values)
    hostmass += list(surv[survey].HOST_LOGMASS.values)
    hostmass_err += list(surv[survey].HOST_LOGMASS_ERR.values)
    redshifts += list(surv[survey].zCMB.values)
    infor += list(stretchevol.Evol2G2M2S.delta(surv[survey].zCMB.values))
    py += list([0 for i in range(len(surv[survey].zCMB.values))])
    lssfr += list([0 for i in range(len(surv[survey].zCMB.values))])
    lssfr_err_d += list([0 for i in range(len(surv[survey].zCMB.values))])
    lssfr_err_u += list([0 for i in range(len(surv[survey].zCMB.values))])

df_full = pd.DataFrame({'survey': names,
                        'stretchs': stretchs,
                        'stretchs_err': stretchs_err,
                        'colors': colors,
                        'colors_err': colors_err,
                        'hostmass': hostmass,
                        'hostmass_err': hostmass_err,
                        'redshifts': redshifts,
                        'infor': infor,
                        'py': py,
                        'lssfr': lssfr,
                        'lssfr_err_d': lssfr_err_d,
                        'lssfr_err_u': lssfr_err_u})
