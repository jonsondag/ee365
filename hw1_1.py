import numpy as np
import pandas as pd
import utils

###########
# problem 1
###########

def print_prescient_mean(p0, p1):
    utils.plot_histogram(pd.Series(np.maximum(p0, p1)), '1.1b prescient revenues', 'num paths with revenue', 'revenue')
    print 'prescient expected revenue', str(np.mean(np.maximum(p0, p1)))


def print_no_knowledge_mean(p0, p1, mu0, mu1, sigma0, sigma1):
    exp_mean0 = np.exp(mu0 + sigma0**2 / 2)
    exp_mean1 = np.exp(mu1 + sigma1**2 / 2)
    if exp_mean0 > exp_mean1:
        p_to_use = p0
    else:
        p_to_use = p1
    utils.plot_histogram(pd.Series(p_to_use), '1.1b no knowledge expected revenue', 'num paths with revenue', 'revenue')
    print 'no knowledge mean', str(np.mean(p_to_use))


def print_partial_knowledge_mean(p0, p1, mu1, sigma1):
    exp_mean1 = np.exp(mu1 + sigma1**2 / 2)
    p_to_use = np.where(p0 > exp_mean1, p0, p1)
    utils.plot_histogram(pd.Series(p_to_use), '1.1b partial knowledge expected revenue', 'num paths with revenue',
                         'revenue')
    print 'partial knowledge revenues', str(np.mean(p_to_use))


def prob_1():
    mu0 = 0.0
    mu1 = 0.1
    sigma0 = 0.4
    sigma1 = 0.4

    num_samples = 1000000
    p0 = np.exp(np.random.normal(mu0, sigma0, num_samples))
    p1 = np.exp(np.random.normal(mu1, sigma1, num_samples))

    utils.label('1.1b')
    print_prescient_mean(p0, p1)
    print_no_knowledge_mean(p0, p1, mu0, mu1, sigma0, sigma1)
    print_partial_knowledge_mean(p0, p1, mu1, sigma1)


if __name__ == '__main__':
    prob_1()