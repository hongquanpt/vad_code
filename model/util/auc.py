import matplotlib.pyplot as plt
import numpy as np


def draw_auc(auc, fpr, tpr, color='red', filepath='AUC.png'):
    auc = auc * 100
    plt.plot(fpr, tpr, linestyle='-', color=color)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve, AUC = %.2f' % auc)
    plt.savefig(filepath, dpi=100)
    plt.close()
    # plt.show()


def draw_auc_manual(fpr, tpr, color='darkorange', filepath='AUC_byhand.png'):
    auc = 100 * np.trapz(tpr, fpr)

    plt.plot(fpr, tpr, linestyle='-', color=color, lw=2, label='ROC curve', clip_on=False)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve, AUC = %.2f' % auc)
    plt.legend(loc="lower right")
    plt.savefig(filepath, dpi=100)
    plt.close()
    # plt.show()
