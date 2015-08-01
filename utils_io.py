import matplotlib.pyplot as plt
import seaborn


######################
# output utils
######################
def label(prob):
    print '--- {0:s} ---'.format(prob)


def plot_histogram(series, title, x_label, y_label):
    f1 = plt.figure()
    ax = series.hist()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    f1.suptitle(title)
    f1.show()
    plt.ion()

