import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import metafeatures
import features

DATA_DIRECTORY = "/media/veracrypt2/data"
#DATA_DIRECTORY = "data"

DIAG_CODES = [
    "D611",
    "D642",
    "D695",
    "E032",
    "E064",
    "E160",
    "E273",
    "G211",
    "G240",
    "G251",
    "G444",
    "G620",
    "I427",
    "I952",
    "L270",
    "L271",
    "M804",
    "M814",
    "N141",
    "O355",
    "R502",
    "T599",
    "T782",
    "T783",
    "T784",
    "T789",
    "T801",
    "T802",
    "T808",
    "T809",
    "T886",
    "T887",
]

path_for = "{0}/{1}-90-raw-measurements.csv".format


def read_frame(diag_code):
    return pd.read_csv(path_for(DATA_DIRECTORY, diag_code))

FILL_NA_VAL = -100000

def get_data_for(diag_code):
    df = read_frame(diag_code)
    xs = df.pivot_table(index=["patientnr", "time"],
                        columns="code",
                        values="value")
    ys = df.groupby("patientnr")["ADE"].mean()
    return xs, ys

def accuracy(y_hat, y_test):
    corr = (y_test == y_hat).sum()
    total = y_hat.shape[0]
    return corr / float(total)

def build_base_model(xs, ys):
    clf = RandomForestClassifier(n_estimators=1, n_jobs=-1, max_features=None)
    clf.fit(xs, ys)
    return clf

def evaluate_model(clf, x_test, y_test):
    y_hat = clf.predict(x_test)
    return accuracy(y_test, y_hat)

def build_and_evaluate(features, ys):
    x_train, x_test, y_train, y_test = train_test_split(features.fillna(FILL_NA_VAL), ys)
    clf = build_base_model(x_train, y_train)
    return evaluate_model(clf, x_test, y_test)


def iter_all_tses(datasets):
    for (basedata, ys) in datasets:
        for tsvar in basedata.columns.values:
            yield basedata, ys, tsvar


def build_meta_model_ys(datasets):
    last_basedata = None
    cache = None
    i = 0
    for (basedata, ys, tsvar) in iter_all_tses(datasets):
        if cache is None or basedata is not last_basedata:
            cache = dict()
            last_basedata = basedata
        dfs = features.df_to_all_reprs(basedata, tsvar, cache)
        yield [build_and_evaluate(df, ys) for df in dfs]
        i += 1
        if i % 100 == 0:
            print("i = {0}".format(i))


def build_meta_model_xs(datasets):
    for (basedata, ys, tsvar) in iter_all_tses(datasets):
        yield metafeatures.extract_meta_features_as_arr(basedata[tsvar])


def build_meta_model(xs, ys):
    clf = RandomForestClassifier(n_estimators=1, n_jobs=-1, max_features=None)
    clf.fit(xs, ys)
    return clf

def meta_build_and_evaluate(xs, ys):
    x_train, x_test, y_train, y_test = train_test_split(xs, ys)
    clf = build_meta_model(x_train, y_train)
    y_hat = clf.predict(x_test)
    mean_squared_error = ((y_hat - y_test) ** 2).mean()
    return mean_squared_error


if __name__ == "__main__":
    TRAINING_FILES = ["T887"]
    data = [get_data_for(x) for x in TRAINING_FILES]
    print("Building X-values for meta-model")
    xs = np.array(list(build_meta_model_xs(data)))
    np.save("meta-model-xs.npy", xs)
    print("Building Y-values for meta-model")
    ys = np.array(list(build_meta_model_ys(data)))
    np.save("meta-model-ys.npy", ys)
    print("Building meta model")
    meta_build_and_evaluate(xs, ys)
