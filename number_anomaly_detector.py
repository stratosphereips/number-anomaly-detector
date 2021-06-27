#!/usr/bin/env python3
# This file is part of the Stratosphere Linux IPS
# See the file 'LICENSE' for copying permission.
# Authors:
# - Sebastian Garcia. eldraco@gmail.com,
#   sebastian.garcia@agents.fel.cvut.cz

import pandas as pd
# from sklearn.model_selection import train_test_split
# from pyod.models import lof
# from pyod.models.abod import ABOD
# from pyod.models.cblof import CBLOF
# from pyod.models.lof import LOF
# from pyod.models.loci import LOCI
# from pyod.models.lscp import LSCP
# from pyod.models.mcd import MCD
# from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
# from pyod.models.sod import SOD
# from pyod.models.so_gaal import SO_GAAL # Needs keras
# from pyod.models.sos import SOS  # Needs keras
# from pyod.models.xgbod import XGBOD # Needs keras
# from pyod.models.knn import KNN   # kNN detector
import argparse
import warnings


# This horrible hack is only to stop sklearn from printing those warnings
def warn(*args, **kwargs):
    pass


warnings.warn = warn


def detect(file, amountanom, realtime):
    """
    Function to apply a very simple anomaly detector
    amountanom: The top number of anomalies we want to print
    realtime: If we want to read the conn.log file in real time (not working)
    """

    # Create a Pandas dataframe from the conn.log
    df = pd.read_csv(file, names=['values'])

    # Replace the rows without data (with '-') with 0.
    df['values'].replace('-', '0', inplace=True)

    # Add the columns from the log file that we know are numbers. This is only for conn.log files.
    X_train = df[['values']]

    # The X_test is where we are going to search for anomalies. In our case, its the same set of data than X_train.
    X_test = X_train

    #################
    # Select a model from below

    # ABOD class for Angle-base Outlier Detection. For an observation, the
    # variance of its weighted cosine scores to all neighbors could be
    # viewed as the outlying score.
    # clf = ABOD()

    # LOF
    # clf = LOF()

    # CBLOF
    # clf = CBLOF()

    # LOCI
    # clf = LOCI()

    # LSCP
    # clf = LSCP()

    # MCD
    # clf = MCD()

    # OCSVM
    # clf = OCSVM()

    # PCA. Good and fast!
    clf = PCA()

    # SOD
    # clf = SOD()

    # SO_GAAL
    # clf = SO_GALL()

    # SOS
    # clf = SOS()

    # XGBOD
    # clf = XGBOD()

    # KNN
    # Good results but slow
    # clf = KNN()
    # clf = KNN(n_neighbors=10)
    #################

    # Fit the model to the train data
    clf.fit(X_train)

    # get the prediction on the test data
    y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)

    y_test_scores = clf.decision_function(X_test)  # outlier scores

    # Convert the ndarrays of scores and predictions to  pandas series
    scores_series = pd.Series(y_test_scores)
    pred_series = pd.Series(y_test_pred)

    # Now use the series to add a new column to the X test
    X_test['score'] = scores_series.values
    X_test['pred'] = pred_series.values

    # Add the score to the df also. So we can show it at the end
    df['score'] = X_test['score']

    # Keep the positive predictions only. That is, keep only what we predict is an anomaly.
    X_test_predicted = X_test[X_test.pred == 1]

    # Keep the top X amount of anomalies
    top10 = X_test_predicted.sort_values(by='score', ascending=False).iloc[:amountanom]

    # Print the results
    # Find the predicted anomalies in the original df dataframe, where the rest of the data is
    df_to_print = df.iloc[top10.index]
    print('\nTop anomalies')

    # Only print some columns, not all, so its easier to read.
    #df_to_print = df_to_print.drop(['values'], axis=1)
    # Dont print index
    print(df_to_print.to_string(index=False))


if __name__ == '__main__':
    print('Simple Number Anomaly Detector. Version: 0.1')
    print('Author: Sebastian Garcia (eldraco@gmail.com)')

    # Parse the parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='Amount of verbosity. This shows more info about the results.', action='store', required=False, type=int)
    parser.add_argument('-e', '--debug', help='Amount of debugging. This shows inner information about the program.', action='store', required=False, type=int)
    parser.add_argument('-f', '--file', help='Path to the input file to read.', required=True)
    parser.add_argument('-a', '--amountanom', help='Amount of anomalies to show.', required=False, default=10, type=int)
    parser.add_argument('-R', '--realtime', help='Read from stdin.', required=False, type=bool, default=False)
    args = parser.parse_args()

    detect(args.file, args.amountanom, args.realtime)
