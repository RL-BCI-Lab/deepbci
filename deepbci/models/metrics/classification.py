from sklearn.metrics import precision_recall_curve, roc_curve, auc

def roc_auc_score(y_true, y_pred, column=None, **kwargs):
    if column is not None:
        if len(y_true.shape) > 1:
            y_true = y_true[:, column]
        if len(y_pred.shape) > 1:
            y_pred = y_pred[:, column]
    fpr, tpr, _ = roc_curve(y_true, y_pred, **kwargs)
    return auc(fpr, tpr)
               
def prc_auc_score(y_true, y_pred, column=None, **kwargs):
    if column is not None:
        if len(y_true.shape) > 1:
            y_true = y_true[:, column]
        if len(y_pred.shape) > 1:
            y_pred = y_pred[:, column]
    ppv, tpr, _ = precision_recall_curve(y_true, y_pred, **kwargs)
    return auc(tpr, ppv)