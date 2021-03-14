import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
        # TODO: Make plots for loss curves and accuracy curves.
        # TODO: You do not have to return the plots.
        # TODO: You can save plots as files by codes here or an interactive way according to your preference.
        # Loss curve
        plt.figure(1)
        plt.plot(train_losses, label='Train Loss', linewidth=1.0, color='blue')
        plt.plot(valid_losses, label='Validation Loss', linewidth=1.0, color='red')
        plt.xlabel('Epoch', fontsize = 16)
        plt.ylabel('Loss', fontsize = 16)
        plt.title('Loss Curve', fontsize = 18)
        plt.legend(loc = 'best')
        plt.savefig('loss_curves.png')
        plt.close(1)

        # Accuracy Curve
        plt.figure(2)
        plt.plot(train_accuracies, label='Train Accuracy', linewidth=1.0, color='blue')
        plt.plot(valid_accuracies, label='Validation Accuracy', linewidth=1.0, color='red')
        plt.xlabel('Epoch', fontsize = 16)
        plt.ylabel('Accuracy', fontsize = 16)
        plt.title('Accuracy Curve', fontsize = 18)
        plt.legend(loc = 'best')
        plt.savefig('accuracy_curves.png')
        plt.close(2)
        pass


def plot_confusion_matrix(results, class_names):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.

        def plot_confusion_matrix(cm, classes,cmap=plt.cm.Blues):
                # The code is borrowed from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
                cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
                fig, ax = plt.subplots()
                im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
                ax.figure.colorbar(im, ax=ax)
                # We want to show all ticks...
                ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
                        xticklabels=classes, yticklabels=classes,
                        title='Normalized confusion matrix',
                        ylabel='True Label',
                        xlabel='Predicted Label')

                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

                # Loop over data dimensions and create text annotations.
                fmt = '.2f'
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                                ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                                        color="white" if cm[i, j] > thresh else "black")
                fig.tight_layout()


        cm = confusion_matrix(list(zip(*results))[0], list(zip(*results))[1])
        np.set_printoptions(precision=2)
        plot_confusion_matrix(cm, classes=class_names)
        plt.savefig("confusion_matrix.png")
        pass
