import os
import numpy as np
import pandas as pd

from mg.stats.conf_perc import conf_quantile, norm_stand


def preprocess(filename, confidence_half_tertile=False):

    d = pd.read_csv(filename)

    fname = os.path.split(filename)[1]

    # CalderTravis_unpub, OHora_unpub_2: confidence ratings < 0 indicated "wrong responses"; here we set these
    # ratings to the lowest >=0 confidence rating
    if fname in ('data_CalderTravis_unpub.csv', 'data_OHora_unpub_2.csv'):
        d.loc[d.Confidence < 0, 'Confidence'] = d[d.Confidence >= 0].min()
    # OHora_*: normalize ratings
    if fname in ('data_OHora_unpub_1.csv', 'data_OHora_unpub_2.csv', 'data_OHora_2017.csv'):
        d.Confidence /= d.Confidence.min()
    # Palser_2018: confidence ratings could exceed the designated maximum value of 99
    if fname == 'data_Palser_2018.csv':
        d.loc[d.Confidence > 99, 'Confidence'] = 99
    # Hainguerlot_2018: convert fractional ratings to integer ratings
    if fname in ('data_Hainguerlot_2018.csv', 'data_Hainguerlot_unpub.csv'):
        d.Confidence *= 10
    # Desender_2016: convert string ratings to integer ratings
    if fname == 'data_Desender_2016.csv':
        d.Confidence = d.Confidence.apply(lambda x: {'diff': 1, 'easy': 2, np.nan: np.nan}[x])
    # Paulewicz_unpub2: create binary responses - response column includes 4 different values,
    # but we can reconstruct binary responses from the Accuracy column
    if fname == 'data_Paulewicz_unpub2.csv':
        d.loc[~d.Response.isna(), 'Response'] = ((d[~d.Response.isna()].Stimulus - 1) == d[~d.Response.isna()].Accuracy) + 1
    # Siedlecka_unpub: account for two different sorts of NaNs
    if fname == 'data_Siedlecka_unpub.csv':
        d.Response = d.Response.astype(float)
    # Sinanaj_2015: convert zeros (standing for 'no response' to NaN)
    if fname == 'data_Sinanaj_2015.csv':
        d.loc[d.Response == 0] = np.nan

    if confidence_half_tertile:
        if len(d[~d.Confidence.isna()].Confidence.unique()) > 2:
            d = conf_quantile(d, 2, sourcename='Confidence', targetname='confidence_half', subj_name='Subj_idx', start_at_1=True, spread=True)
        else:
            # ratings should always begin at 1:
            d['confidence_half'] = d.Confidence.values
            d.confidence_half -= (d.confidence_half.min() - 1)

        if len(d[~d.Confidence.isna()].Confidence.unique()) > 3:
            d = conf_quantile(d, 3, sourcename='Confidence', targetname='confidence_tertile', subj_name='Subj_idx', start_at_1=True, spread=True)
        else:
            # ratings should always begin at 1:
            d['confidence_tertile'] = d.Confidence.values
            d.confidence_tertile -= (d.confidence_tertile.min() - 1)

    d['confidence'] = d.Confidence.values

    if len(d[~d.Confidence.isna()].Confidence.unique()) > 6:
        d = conf_quantile(d, 6, sourcename='Confidence', targetname='Confidence', subj_name='Subj_idx', start_at_1=True)
    else:
        # ratings should always begin at 1:
        d.Confidence -= (d.Confidence.min() - 1)

    if fname in ('data_Haddara_unpub.csv', 'data_Massoni_2017.csv'):
        d = d.rename(columns=(dict(Feedback='feedback_given')))
        d = d.astype(dict(feedback_given='bool'))
    elif fname == 'data_Kantner_2010.csv':
        d['feedback_given'] = d.Condition.apply(lambda x: 'no feedback' not in x)
    elif fname == 'data_Lebreton_2018.csv':
        d['feedback_given'] = (d.feeback_performance != 0)
    elif fname == 'data_Rouault_2019.csv':
        d['feedback_given'] = d.Confidence.isna()
    elif fname == 'data_Siedlecka_2019.csv':
        d['feedback_given'] = d.Condition.apply(lambda x: dict(nofeedback=False, feedback=True)[x])

    d['confidence_meta'] = d.Confidence.values
    d = conf_quantile(d, 4, sourcename='confidence', targetname='confidence_quartile', subj_name='Subj_idx', start_at_1=False, spread=True)
    d = norm_stand(d, sourcename_rt=False)

    return d


def preprocess2(z):
    zsubject = 0
    for id in range(len(z.study_id.unique())):
        for i, s in enumerate(sorted(z[z.study_id == id].subject.unique())):
            z.loc[(z.study_id == id) & (z.subject == i), 'zsubject'] = zsubject
            zsubject += 1
    return z

def meta_info(z):

    HOME = os.path.expanduser('~')
    path_project = os.path.join(HOME, 'Dropbox/data/confidencedb/')
    db = pd.read_excel(os.path.join(path_project, 'Database_Information.xlsx'))

    dbvars = ['nsubjects', 'min_trials_per_subject', 'nratings', 'continuous', 'conf_dec_simu', 'trial_feedback', 'block_feedback']
    for name in z.study_name.unique():
        category = 'mixed' if 'Mixed' in db[db.Name_in_database == name].Category.values[0] else db[db.Name_in_database == name].Category.values[0].lower().strip()
        z.loc[z.study_name == name, 'category'] = category
        z.loc[z.study_name == name, 'category_id'] = dict(perception=0, memory=1, cognitive=2, motor=3, mixed=4)[category]
        feedback_given = int(db[db.Name_in_database == name].Feedback.values[0])
        if feedback_given != 2:  # if feedback_given 2, we already included the feedback condition for each subject
            z.loc[z.study_name == name, 'feedback_given'] = feedback_given
        z.loc[z.study_name == name, 'trial_feedback'] = int(db[db.Name_in_database == name].Feedback.values[0])
        for var in dbvars:
            z.loc[z.study_name == name, var] = db.loc[db.Name_in_database == name, var].values[0]

    z['any_feedback'] = (z['trial_feedback'] == 1) | (z['block_feedback'] == 1)
    z['perception'] = z.category_id == 0
    z['memory'] = z.category_id == 1
    z['cognitive'] = z.category_id == 2
    z['motor'] = z.category_id == 3
    z['mixed'] = z.category_id == 4

    return z
