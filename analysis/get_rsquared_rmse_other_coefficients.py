# @https://github.com/tatwan/Linear-Regression-Implementation-in-Python/blob/master/Linear_Regression_Python.ipynb
# @https://stackabuse.com/calculating-pearson-correlation-coefficient-in-python-with-numpy/
# @https://realpython.com/numpy-scipy-pandas-correlation-python/
# @https://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy

import sys
import argparse
import warnings

from tqdm import tqdm
from espnet.utils.cli_utils import strtobool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, linregress
import statsmodels.api as sml
from statsmodels import tools

mapping_dict = {
    0: 0,
    'A1': 1,
    'A1+': 2,
    'A2': 3,
    'A2+': 4,
    'B1': 5,
    'B1+': 6,
    'B2': 7
}

analysis_types_list = ['rmse', 'rsquared', 'accuracy']
accuracy_within = [0.5, 1]

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

def get_r2_numpy_corrcoef(x, y):
    return np.corrcoef(x, y)[0, 1]**2

def nullable_string(val):
    if val.lower() == 'none':
        return None
    return val

def argparse_function():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv_file_path",
                        default='data/trn/text_list',
                        type=str)

    for item in analysis_types_list:
        parser.add_argument("--output_{}_file_path",
                            default=None,
                            type=nullable_string)

    args = parser.parse_args()

    return args

def accuracy_within_margin(score_predictions, score_target, margin):
    """ Returns the percentage of predicted scores that are within the provided margin from the target score. """
    return np.sum(
        np.where(
            np.absolute(score_predictions - score_target) <= margin,
            np.ones(len(score_predictions)),
            np.zeros(len(score_predictions)))).item() / len(score_predictions) * 100

def get_rmse_numpy(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def results_summary_to_dataframe(results):
    '''take the result of an statsmodel results table and transforms it into a dataframe'''
    pvals = results.pvalues
    coeff = results.params
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]

    results_df = pd.DataFrame({"pvals":pvals,
                               "coeff":coeff,
                               "conf_lower":conf_lower,
                               "conf_higher":conf_higher
                                })

    #Reordering...
    results_df = results_df[["coeff","pvals","conf_lower","conf_higher"]]
    return results_df


if __name__ == '__main__':

    # warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()

    # variables
    args = argparse_function()
    input_csv_file_path = args.input_csv_file_path
    output_problem_utts_list_file_path = args.output_problem_utts_list_file_path

    # read file
    dataset = pd.read_csv(input_csv_file_path)

    # preprocess
    Y = dataset[['level']]
    X = dataset.drop(['level'],  axis=1)
    columns = X.columns

    items_problem_dict = {}
    items_rmse_dict = {}
    items_accuracy_dict = {}
    items_rsquared_dict = {}

    # OLS regression model for each kind of feature
    for item in columns:

        input_x = dataset[[item]]

        ## R-squared value
        # @See: https://www.adamsmith.haus/python/answers/how-to-calculate-r-squared-with-numpy-in-python
        minor_ols = sml.OLS(endog=Y, exog=input_x).fit() # Train a new regression model
        predicted_y = minor_ols.predict(input_x).to_numpy()
        target_y    = np.squeeze(Y.to_numpy(), axis=1)
        if np.amax(predicted_y.to_numpy()) == np.amin(predicted_y.to_numpy()):
            items_problem_dict[item] = input_x.to_numpy()
        else:
            rsquared_value = get_r2_numpy_corrcoef(predicted_y, target_y)
            items_rsquared_dict.setdefault(item, rsquared_value)
            # plt.subplots(figsize=(10,8))
            # sns.regplot(x=item, y='level', data=dataset, color='g')

        ## R-MSE value
        rmse_value = get_rmse_numpy(predicted_y, target_y)
        items_rmse_dict.setdefault(item, rmse_value)

        ## Accuracy value
        for within in accuracy_within:
            accuracy_value = accuracy_within_margin(predicted_y, target_y, within)
            items_accuracy_dict.setdefault(within, {}).setdefault(item, accuracy_value)

    # save files
    for item in analysis_types_list:
        file_path = getattr(args, "output_{}_file_path".format(item))
        if file_path is not None:
            with open(file_path, 'w') as f:
                for item, rmse_value in items_rmse_dict.items():
                    f.write("{} {}\n".format(item, rmse_value))

    if items_problem_dict:
        print(items_problem_dict)
