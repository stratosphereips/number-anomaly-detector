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
import sys
import fileinput



# This horrible hack is only to stop sklearn from printing those warnings
def warn(*args, **kwargs):
    pass


warnings.warn = warn


def detect(input_data, amountanom, window_size):
    """
    Parent function to deal with real time or not

    input_data: can be a file to open of STDIN
    """
    if args.verbose:
        print('Detecting')
    if input_data:
        if args.verbose:
            print('By file')
        # Create a Pandas dataframe from the conn.log
        df = pd.read_csv(input_data, names=['values'])
        detect_numbers(df, amountanom)
    else:
        if args.verbose:
            print('Realtime')
        # Read in groups of 'window_size' width and train and test on them
        read_lines = 0
        lines = []
        try:
            for line in iter(sys.stdin.readline, b''):
                if line:
                    line = line.strip()
                    lines.append(line)
                    read_lines += 1
                    if read_lines == window_size:
                        print(f'Read new numbers. Processing...')
                        df = pd.DataFrame(lines, columns=['values'])
                        detect_numbers(df, amountanom)
                        read_lines = 0
                else:
                    break
            # Capture the last batch
            df = pd.DataFrame(lines, columns=['values'])
            detect_numbers(df, amountanom)
        except KeyboardInterrupt:
            sys.stdout.flush()

def detect_numbers(df, amountanom):
    """
    Function to apply a very simple anomaly detector to a set of numbers given as a pandas dataframe
    amountanom: The top number of anomalies we want to print
    df: input dataframe with numbers
    """

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
    parser.add_argument('-f', '--file', help='Path to the input file to read.', required=False)
    parser.add_argument('-a', '--amountanom', help='Amount of anomalies to show.', required=False, default=10, type=int)
    parser.add_argument('-w', '--window_size', help='Width of the groups of numbers to read and detect if using STDIN.', required=False, type=bool, default=10)
    args = parser.parse_args()

    detect(args.file, args.amountanom, window_size=args.window_size)
