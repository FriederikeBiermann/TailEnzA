import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import numpy as np
def plot_confusion_matrices(enzyme, classifier, x_test, y_test, BGC_types):
    np.set_printoptions(precision=2)
    class_names=map(str, BGC_types)
    
    # Plot confusion matrixes
    titles_options = [("Confusion matrix, without normalization "+enzyme, None),
                      ("Normalized confusion matrix "+enzyme, 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, x_test, y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)
        #disp.plot()
        for labels in disp.text_.ravel():
            labels.set_fontsize(10)
        plt.figure(figsize=(15,8))
        plt.savefig(("confusionmatrix_"+str(normalize)+"_"+enzyme+"_.png"), format="png")
        plt.show() 
