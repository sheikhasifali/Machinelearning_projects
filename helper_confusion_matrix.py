import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def adjusted_classes(pred_prob, threshold):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= threshold else 0 for y in pred_prob]

def conf_matrix(ytest,lr_pred_prb,threshold=0.5):
    pred_adj = adjusted_classes(lr_pred_prb, threshold)
    tn, fp, fn, tp = confusion_matrix(ytest, pred_adj).ravel()
   
    #confusion matrix
    print(pd.DataFrame({"pred_Survived":[tp,fp],"pred_Not Survived":[fn,tn]},index=["Survived","Not Survived"]))
    
    #accuracy
    print("Accuracy: %0.3f"%((tp+tn)/(tn+fp+fn+tp)*100))
    
    #precision
    precision_1 = tp / (tp + fp)
    print("Precision : %0.3f"% (precision_1*100))
       
    #recall
    recall_1 = tp / (tp + fn)
    print("Recall: %0.3f"%(recall_1*100))
    
    #f1 score
    f1_1 = (2 * precision_1 * recall_1) / (precision_1 + recall_1)
    print("F1 Score : %0.3f"%(f1_1*100))

    tpr = tp / (fn + tp)
    fpr = fp / (fp + tn)
    print("TPR: %0.3f"%(tpr*100)," FPR: %0.3f"%(fpr*100))