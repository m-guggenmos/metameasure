import matlab.engine
eng = matlab.engine.start_matlab()
import numpy as np
from collections import namedtuple

# Python API for Steve Fleming's hmeta-d' (https://github.com/metacoglab/HMeta-d)
# Matthias Guggenmos, 2021


def type2_SDT_MLE_group(stimID, response, rating, nRatings, cellpadding=0):
    # out = type2_SDT(input)
    #
    # Given data from an experiment where an observer discriminates between two
    # stimulus alternatives on every trial and provides confidence ratings,
    # provides a type 2 SDT analysis of the data.
    #
    # The function estimates the parameters of the unequal variance SDT model,
    # and uses those estimates to find a maximum likelihood estimate of
    # meta-da.
    #
    # INPUTS
    #
    # format of the input may be either:
    #
    # 1) stimID, response, rating, nRatings, (cellpadding), (equalVariance)
    #    where each of the first 3 inputs is a 1xN vector describing the outcome
    #    of N trials. Contents of input should be as follows.
    #
    #    stimID   : 0=S1 stimulus presented, 1=S2 stimulus presented
    #    response : 0=subject responded S1, 1=subject responded S2
    #    rating   : values ranges from 1 to m where 1 is the lowest rating
    #               and m is the highest.
    #
    #               All trials where any of these prescribed ranges of values
    #               are violated are omitted from analysis.
    #
    #    nRatings : the number of ratings available to the subject (e.g. for a
    #               confidence scale of 1-4, nRatings=4).
    #    cellpadding : if any data cells (e.g. high confidence "S2" responses)
    #               are empty, then the value of cellpadding will be added
    #               to every data cell. If not specified, default = 1/(2*nRatings)
    #    equalVariance : if 1, force analysis to use the equal variance SDT
    #               model. If 0, use an estimate of s = sd(S1) / sd(S2) where
    #               s is the slope of the zROC data (estimated using MLE).
    #               If not specified, default = 0.
    #
    # 2) nR_S1, nR_S2, (cellpadding), (equalVariance)
    #    where these are vectors containing the total number of responses in
    #    each response category, conditional on presentation of S1 and S2.
    #    size of each array is 2*nRatings, where each element corresponds to a
    #    count of responses in each response category. Response categories are
    #    ordered as follows:
    #    highest conf "S1" ... lowest conf "S1", lowest conf "S2", ... highest conf "S2"
    #
    #    e.g. if nR_S1 = [100 50 20 10 5 1], then when stimulus S1 was
    #    presented, the subject had the following response counts:
    #    responded S1, rating=3 : 100 times
    #    responded S1, rating=2 : 50 times
    #    responded S1, rating=1 : 20 times
    #    responded S2, rating=1 : 10 times
    #    responded S2, rating=2 : 5 times
    #    responded S2, rating=3 : 1 time
    #
    #    cellpadding and equalVariance are defined as above.
    #
    #
    #
    #
    # OUTPUTS
    #
    # out.d_a       : d_a for input data. If s=1, d_a = d'
    # out.meta_d_a  : meta_d_a for input data
    # out.M_ratio   : meta_d_a / d_a; measure of metacognitive efficiency
    # out.M_diff    : meta_d_a - d_a; measure of metacognitive efficiency
    # out.c_a       : criterion c_a for input data. If s=1, c_a = c.
    # out.cprime    : relative criterion used for type 2 estimates. c' = c_a / d_a
    # out.s         : ratio of evidence distribution standard deviations assumed for the analysis.
    # out.type2_fit : output of fit_meta_d_MLE for the type 2 SDT fit.

    # 9/24/10 - bm - fixed program-crashing bug for (nR_S1, nR_S2) input
    # 9/7/10 - bm - wrote it

    ## parse inputs

    if cellpadding is None:
        cellpadding = 1 / (2 * nRatings)

    nsubj = len(stimID)
    nR_S1, nR_S2 = np.full((nsubj, nRatings * 2), np.nan), np.full((nsubj, nRatings * 2), np.nan)

    for s in range(nsubj):
        # filter bad trials
        f = ((stimID[s] == 0) | (stimID[s] == 1)) & ((response[s] == 0) | (response[s] == 1)) & ((rating[s] >= 1) & (rating[s] <= nRatings))
        stimID[s] = stimID[s][f]
        response[s] = response[s][f]
        rating[s] = rating[s][f]

        # get tallies of "S1" rating responses for S1 and S2 stim
        for i in range(nRatings):
            nR_S1[s, i] = np.sum((stimID[s] == 0) & (response[s] == 0) & (rating[s] == nRatings - i))
            nR_S2[s, i] = np.sum((stimID[s] == 1) & (response[s] == 0) & (rating[s] == nRatings - i))

        # get tallies of "S2" rating responses for S1 and S2 stim
        for i in range(nRatings):
            nR_S1[s, i + nRatings] = np.sum((stimID[s] == 0) & (response[s] == 1) & (rating[s] == i + 1))
            nR_S2[s, i + nRatings] = np.sum((stimID[s] == 1) & (response[s] == 1) & (rating[s] == i + 1))

        if np.any(nR_S1[s] == 0) | np.any(nR_S2[s] == 0):
            nR_S1[s] = nR_S1[s] + cellpadding
            nR_S2[s] = nR_S2[s] + cellpadding

    fit = eng.fit_meta_d_mcmc_group([eng.cell2mat(v.tolist()) for v in nR_S1], [eng.cell2mat(v.tolist()) for v in nR_S2])

    nt = namedtuple('x', fit.keys())(*fit.values())

    return nt



def type2_SDT_MLE_groupCorr(stimID_task1, response_task1, rating_task1, stimID_task2, response_task2, rating_task2,
                            nRatings, cellpadding=0):
    # out = type2_SDT(input)
    #
    # Given data from an experiment where an observer discriminates between two
    # stimulus alternatives on every trial and provides confidence ratings,
    # provides a type 2 SDT analysis of the data.
    #
    # The function estimates the parameters of the unequal variance SDT model,
    # and uses those estimates to find a maximum likelihood estimate of
    # meta-da.
    #
    # INPUTS
    #
    # format of the input may be either:
    #
    # 1) stimID, response, rating, nRatings, (cellpadding), (equalVariance)
    #    where each of the first 3 inputs is a 1xN vector describing the outcome
    #    of N trials. Contents of input should be as follows.
    #
    #    stimID   : 0=S1 stimulus presented, 1=S2 stimulus presented
    #    response : 0=subject responded S1, 1=subject responded S2
    #    rating   : values ranges from 1 to m where 1 is the lowest rating
    #               and m is the highest.
    #
    #               All trials where any of these prescribed ranges of values
    #               are violated are omitted from analysis.
    #
    #    nRatings : the number of ratings available to the subject (e.g. for a
    #               confidence scale of 1-4, nRatings=4).
    #    cellpadding : if any data cells (e.g. high confidence "S2" responses)
    #               are empty, then the value of cellpadding will be added
    #               to every data cell. If not specified, default = 1/(2*nRatings)
    #    equalVariance : if 1, force analysis to use the equal variance SDT
    #               model. If 0, use an estimate of s = sd(S1) / sd(S2) where
    #               s is the slope of the zROC data (estimated using MLE).
    #               If not specified, default = 0.
    #
    # 2) nR_S1_task1, nR_S2_task1, (cellpadding), (equalVariance)
    #    where these are vectors containing the total number of responses in
    #    each response category, conditional on presentation of S1 and S2.
    #    size of each array is 2*nRatings, where each element corresponds to a
    #    count of responses in each response category. Response categories are
    #    ordered as follows:
    #    highest conf "S1" ... lowest conf "S1", lowest conf "S2", ... highest conf "S2"
    #
    #    e.g. if nR_S1_task1 = [100 50 20 10 5 1], then when stimulus S1 was
    #    presented, the subject had the following response counts:
    #    responded S1, rating=3 : 100 times
    #    responded S1, rating=2 : 50 times
    #    responded S1, rating=1 : 20 times
    #    responded S2, rating=1 : 10 times
    #    responded S2, rating=2 : 5 times
    #    responded S2, rating=3 : 1 time
    #
    #    cellpadding and equalVariance are defined as above.
    #
    #
    #
    #
    # OUTPUTS
    #
    # out.d_a       : d_a for input data. If s=1, d_a = d'
    # out.meta_d_a  : meta_d_a for input data
    # out.M_ratio   : meta_d_a / d_a; measure of metacognitive efficiency
    # out.M_diff    : meta_d_a - d_a; measure of metacognitive efficiency
    # out.c_a       : criterion c_a for input data. If s=1, c_a = c.
    # out.cprime    : relative criterion used for type 2 estimates. c' = c_a / d_a
    # out.s         : ratio of evidence distribution standard deviations assumed for the analysis.
    # out.type2_fit : output of fit_meta_d_MLE for the type 2 SDT fit.

    # 9/24/10 - bm - fixed program-crashing bug for (nR_S1_task1, nR_S2_task1) input
    # 9/7/10 - bm - wrote it

    ## parse inputs

    if cellpadding is None:
        cellpadding = 1 / (2 * nRatings)

    nsubj = len(stimID_task1)
    nR_S1_task1, nR_S2_task1 = np.full((nsubj, nRatings * 2), np.nan), np.full((nsubj, nRatings * 2), np.nan)
    nR_S1_task2, nR_S2_task2 = np.full((nsubj, nRatings * 2), np.nan), np.full((nsubj, nRatings * 2), np.nan)

    for s in range(nsubj):
        # filter bad trials
        f = ((stimID_task1[s] == 0) | (stimID_task1[s] == 1)) & ((response_task1[s] == 0) | (response_task1[s] == 1)) & ((rating_task1[s] >= 1) & (rating_task1[s] <= nRatings))
        stimID_task1[s] = stimID_task1[s][f]
        response_task1[s] = response_task1[s][f]
        rating_task1[s] = rating_task1[s][f]
        # get tallies of "S1" rating responses for S1 and S2 stim
        for i in range(nRatings):
            nR_S1_task1[s, i] = np.sum((stimID_task1[s] == 0) & (response_task1[s] == 0) & (rating_task1[s] == nRatings - i))
            nR_S2_task1[s, i] = np.sum((stimID_task1[s] == 1) & (response_task1[s] == 0) & (rating_task1[s] == nRatings - i))
        # get tallies of "S2" rating responses for S1 and S2 stim
        for i in range(nRatings):
            nR_S1_task1[s, i + nRatings] = np.sum((stimID_task1[s] == 0) & (response_task1[s] == 1) & (rating_task1[s] == i + 1))
            nR_S2_task1[s, i + nRatings] = np.sum((stimID_task1[s] == 1) & (response_task1[s] == 1) & (rating_task1[s] == i + 1))
        if np.any(nR_S1_task1[s] == 0) | np.any(nR_S2_task1[s] == 0):
            nR_S1_task1[s] = nR_S1_task1[s] + cellpadding
            nR_S2_task1[s] = nR_S2_task1[s] + cellpadding

        # filter bad trials
        f = ((stimID_task2[s] == 0) | (stimID_task2[s] == 1)) & ((response_task2[s] == 0) | (response_task2[s] == 1)) & ((rating_task2[s] >= 1) & (rating_task2[s] <= nRatings))
        stimID_task2[s] = stimID_task2[s][f]
        response_task2[s] = response_task2[s][f]
        rating_task2[s] = rating_task2[s][f]
        # get tallies of "S1" rating responses for S1 and S2 stim
        for i in range(nRatings):
            nR_S1_task2[s, i] = np.sum((stimID_task2[s] == 0) & (response_task2[s] == 0) & (rating_task2[s] == nRatings - i))
            nR_S2_task2[s, i] = np.sum((stimID_task2[s] == 1) & (response_task2[s] == 0) & (rating_task2[s] == nRatings - i))
        # get tallies of "S2" rating responses for S1 and S2 stim
        for i in range(nRatings):
            nR_S1_task2[s, i + nRatings] = np.sum((stimID_task2[s] == 0) & (response_task2[s] == 1) & (rating_task2[s] == i + 1))
            nR_S2_task2[s, i + nRatings] = np.sum((stimID_task2[s] == 1) & (response_task2[s] == 1) & (rating_task2[s] == i + 1))
        if np.any(nR_S1_task2[s] == 0) | np.any(nR_S2_task2[s] == 0):
            nR_S1_task2[s] = nR_S1_task2[s] + cellpadding
            nR_S2_task2[s] = nR_S2_task2[s] + cellpadding

    fit = eng.fit_meta_d_mcmc_groupCorr2([eng.cell2mat(v.tolist()) for v in nR_S1_task1], [eng.cell2mat(v.tolist()) for v in nR_S2_task1], [eng.cell2mat(v.tolist()) for v in nR_S1_task2], [eng.cell2mat(v.tolist()) for v in nR_S2_task2])
    # fit = fit_meta_d_MLE(nR_S1_task1, nR_S2_task1, s)
    # fit = eng.fit_meta_d_mcmc(nR_S1_task1, nR_S2_task1, s)

    nt = namedtuple('x', fit.keys())(*fit.values())

    return nt
