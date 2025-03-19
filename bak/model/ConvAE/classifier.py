from  sklearn.svm import LinearSVC
from  sklearn.metrics  import  accuracy_score , roc_curve , auc , f1_score, confusion_matrix, classification_report, roc_auc_score

import numpy as np
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC.fit
# anomaly == 1, normal == -1
def svm(X_train, Y_train, X_test, Y_test):
   
    X_train, X_test = normalize_data(X_train, X_test, "standard")
    if Y_train.shape[1] > 1:
        Y_train = np.argmax(Y_train, axis=1)
        Y_test = np.argmax(Y_test, axis=1)
   
    clf = LinearSVC (random_state = 0)
    clf.fit(X_train, Y_train)
    
    start = time.time()
    y_pred = clf.predict(X_test)
    end = time.time()
    elapsed = (end - start)/float (len(X_test))
    
    #acc= accuracy_score(Y_test, y_pred)
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
    # pos_label = 1 => anomaly
    fpr_vot , tpr_vot , _ = roc_curve(Y_test , y_pred , pos_label =1,  drop_intermediate=False)
    roc_auc_vot = auc(fpr_vot , tpr_vot)
    
    print ("SVM")
    print('The auc is {} '.format(roc_auc_vot))
    return roc_auc_vot,elapsed