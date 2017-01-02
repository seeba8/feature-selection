import pandas as pd
import numpy as np


def variance(ts):
    return ts.var()


def mean(ts):
    return ts.mean()


def median(ts):
    return ts.median()


def count(ts):
    return ts.unstack().count(axis=1).mean()


def existence(ts):
    us = ts.unstack()
    return (us.count(axis=1) > 0).sum() / us.shape[0]


def ftest(ts):
    mtx = np.array(ts.unstack())
    rowmeans = np.nanmean(mtx, axis=1)
    ssw = np.nan_to_num((mtx.T - rowmeans) ** 2).sum()
    return ts.var() / ssw


META_FEATURES = {
    "variance": variance,
    "ftest": ftest,
    "mean": mean,
    "median": median,
    "count": count,
    "existence": existence,
}



def extract_meta_features(timeseries):
    return {k: META_FEATURES[k](timeseries) for k in META_FEATURES}


def extract_meta_features_as_arr(timeseries):
    metas = extract_meta_features(timeseries)
    return [metas[k] for k in sorted(metas)]


def seq_of_timeseries_variable(df, cols):
    for col in cols:
        d = extract_meta_features(df[col])
        d["name"] = col
        yield d


def compute_for(df, cols):
    return pd.DataFrame(seq_of_timeseries_variable(df, cols))
