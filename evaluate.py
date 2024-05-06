# Data handling
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay


def count_scores(confusion_matrix, model_name_str):
    '''
    returns a dataframe with scores of the model performing
    '''
    TN, FP, FN, TP = confusion_matrix.ravel()
    ALL = TP + FP + FN + TN

    accuracy = round((TP + TN)/ALL, 2)
    true_positive_rate = sensitivity = recall = power = round(TP/(TP+FN), 2)
    false_positive_rate = false_alarm_ratio = fallout = round(FP/(FP+TN), 2)
    true_negative_rate = specificity = selectivity = round(TN/(TN+FP), 2)
    false_negative_rate = miss_rate = round(FN/(FN+TP), 2)
    precision = PPV = round(TP/(TP+FP), 2)
    f1_score = round(2*(precision*recall)/(precision+recall), 2)
    support_pos = int(TP + FN)
    support_neg = int(FP + TN)
    
    rates = ['Accuracy', 
         'True Positive Rate /Recall', 
         'False Positive Rate', 
         'True Negative Rate', 
         'False Negative Rate',
         'Precision', 
         'F1 Score',
         'Support Positive',
         'Support Negative']
    scores = pd.Series([accuracy, true_positive_rate, false_positive_rate, true_negative_rate, 
                        false_negative_rate, precision, f1_score, support_pos, support_neg])
    
    return pd.DataFrame({'Score Name':rates, model_name_str:scores})

def display_cm(confusion_matrix):
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    disp.plot()
