import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn
import utils
import utils_mdp


def get_price_pdf(prices, mean_price, std_dev_price):
    dist = scipy.stats.lognorm([std_dev_price], mean_price)
    pdf = dist.pdf(prices)
    return pdf / pdf.sum()


def get_f(n, m, p1, p2):
    f = np.zeros([n, m, p1, p2])
    for x_val in range(n):
        for u_val in range(m):
            for w1_val in range(p1):
                for w2_val in range(p2):
                    if u_val == 1 and w2_val == 1 and x_val > 0:
                        f[x_val, u_val, w1_val, w2_val] = x_val - 1  # we offered to sell and there was a buyer
                    else:
                        f[x_val, u_val, w1_val, w2_val] = x_val
                    if f[x_val, u_val, w1_val, w2_val] < 0:
                        f[x_val, u_val, w1_val, w2_val] = 0
    return f


def get_g(n, m, p1, p2, prices):
    g = np.zeros([n, m, p1, p2])
    for x_val in range(n):
        for u_val in range(m):
            for w1_val in range(p1):
                for w2_val in range(p2):
                    if x_val > 0 and u_val == 1 and w2_val == 1:
                        g[x_val, u_val, w1_val, w2_val] = prices[w1_val]
    return g


def plot_pol_info_pat(problem_id, pol, times):
    fig, axes = plt.subplots(nrows=len(times), ncols=1)
    for idx, time in enumerate(times):
        df = pd.DataFrame(pol[:, :, time])
        df.plot(title='{0:s}: Whether to sell, depending on current price; time = {1:d}'.format(problem_id, time),
                     ax=axes[idx])
        axes[idx].set_xlabel('Number of stocks in inventory')
        axes[idx].set_ylabel('1->sell; 0->hold')
    plt.ion()


def plot_val(problem_id, v, times):
    df = pd.DataFrame(v[:, times], columns=times)
    ax = df.plot(title='{0:s}: Value function, depending on the current time'.format(problem_id))
    ax.set_xlabel('Number of stocks in inventory')
    ax.set_ylabel('Value function')
    plt.ion()


def get_prices_modified(prices, price_pdf):
    expected_price = np.dot(prices, price_pdf)
    prices_modified = prices.copy()
    prices_modified[np.where(prices_modified < expected_price)] = -1  # set to -1 so we won't choose to sell these cases
    return prices_modified


def get_initial_state(n):
    initial_state = np.zeros(n)
    initial_state[-1:] = 1.0
    return initial_state


def print_probability_unsold(initial_state, P, T, policy_name):
    curr_state = initial_state
    for time in range(T):
        curr_state = np.dot(curr_state, P[:, :, time])
    prob_unsold = np.sum(curr_state[1:])
    print 'probability we have unsold stocks after T, policy ' + policy_name + ':', str(prob_unsold)
    return prob_unsold


def prob_2():
    plt.show()
    prices = np.arange(0.6, 2.1, 0.1)
    mean_price = 0
    std_dev_price = 0.2
    price_pdf = get_price_pdf(prices, mean_price, std_dev_price)
    buy_not_buy = np.array([0.6, 0.4])
    T = 50
    S = 10  # number of stocks
    n = S + 1  # we can hold 0, ..., S stocks, at one of len(prices) prices
    m = 2  # we can not offer to sell (0), or offer to sell (1)
    p1 = len(prices)  # 15 different prices
    p2 = 2  # there may not be a buyer (0), or there may be a buyer (1)
    f = get_f(n, m, p1, p2)
    g = get_g(n, m, p1, p2, prices)

    # part b
    g_final = np.zeros(n)
    pol, v = utils_mdp.value_info_pat(f, -g, -g_final, price_pdf, buy_not_buy, T)  # pass -g, -g_final since we are maximizing revenue
    v = -v
    utils.label('5.2b')
    print 'expected revenue, optimal policy' + ':', str(v[-1, 0])
    plot_pol_info_pat('5.2b', pol, [0, 20, 40, 45])
    plot_val('5.2b', v, [0, 45, 48, 49, 50])

    # part c
    prices_modified = get_prices_modified(prices, price_pdf)
    g_modified = get_g(n, m, p1, p2, prices_modified)
    pol_mod, v_mod = utils_mdp.value_info_pat(f, -g_modified, -g_final, price_pdf, buy_not_buy, T)
    v_mod = -v_mod
    utils.label('5.2c')
    print 'expected revenue, threshold policy' + ':', str(v_mod[-1, 0])
    plot_pol_info_pat('5.2c', pol_mod, [0, 20, 40, 45])
    plot_val('5.2c', v_mod, [0, 45, 48, 49, 50])

    # part d
    fcl, gcl = utils_mdp.cloop_info_pat(f, g, pol, buy_not_buy)
    P = utils_mdp.ftop_info_pat(fcl, price_pdf, buy_not_buy)
    initial_state = get_initial_state(n)
    print_probability_unsold(initial_state, P, T, 'b')
    fcl_mod, gcl_mod = utils_mdp.cloop_info_pat(f, g_modified, pol_mod, buy_not_buy)
    P_mod = utils_mdp.ftop_info_pat(fcl_mod, price_pdf, buy_not_buy)
    initial_state = get_initial_state(n)
    print_probability_unsold(initial_state, P_mod, T, 'c')

    # part e
    UNSOLD_PENALTY = -100
    g_final_mod = UNSOLD_PENALTY * np.ones([n])  # add a penalty for unsold stocks, to incentivize selling
    g_final_mod[0] = 0.
    pol_e, v_e = utils_mdp.value_info_pat(f, -g, -g_final_mod, price_pdf, buy_not_buy, T)
    v_e = -v_e
    utils.label('5.2e')
    fcl_e, gcl_e = utils_mdp.cloop_info_pat(f, g, pol_e, buy_not_buy)
    P_e = utils_mdp.ftop_info_pat(fcl_e, price_pdf, buy_not_buy)
    prob_unsold = print_probability_unsold(initial_state, P_e, T, 'e')
    print 'expected revenue, policy ' + 'e' + ':', str(v_e[-1, 0] - prob_unsold * UNSOLD_PENALTY)  # add back the penalty times the probability we have unsold stocks


if __name__ == '__main__':
    prob_2()
