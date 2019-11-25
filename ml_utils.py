# Helper functions for the pipeline (creating many ML classifiers & their predictions)
from matplotlib import pyplot as plt
import seaborn
import pickle
import re
import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import cohen_kappa_score, precision_score, accuracy_score, f1_score, recall_score, mean_squared_error, precision_recall_fscore_support
from io import StringIO

from scipy.sparse import csr_matrix,vstack
from random import sample

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from statsmodels.stats.weightstats import DescrStatsW


def F1_positive_class(true_positive, false_positive, true_negative, false_negative):
    precision = (1.0 * true_positive / (true_positive + false_positive))
    recall = (1.0 * true_positive / (true_positive + false_negative))
    f1 = (2 * precision * recall) / float(precision + recall)
    print("Precision: " + str(precision) + " Recall: " + str(recall) + " F1: " + str(f1))
