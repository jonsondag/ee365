import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import problem_data
import seaborn
import utils
import utils_mdp


def get_f(n, m, p):
    f = np.zeros([n, m, p])
    for x_val in range(n):
        for u_val in range(m):
            for w_val in range(p):
                next_val = x_val - w_val + u_val
                if next_val < 0:
                    next_val = 0
                if next_val > n - 2:
                    next_val = n - 1  # dummy state
                if x_val == n - 1:
                    f[x_val, u_val, w_val] = n - 1
                else:
                    f[x_val, u_val, w_val] = next_val
    return f


def get_g_order(n, m, p_fixed, p_whole, p_disc, u_disc):
    g_order = np.zeros([n, m])
    for x_val in range(n):
        for u_val in range(m):
            if x_val == n - 1:
                g_order[x_val, u_val] = 10000
            elif u_val == 0:
                g_order[x_val, u_val] = 0
            elif 1 <= u_val <= u_disc:
                g_order[x_val, u_val] = p_fixed + p_whole * u_val
            else:
                g_order[x_val, u_val] = p_fixed + p_whole * u_disc + p_disc * (u_val - u_disc)
    return g_order


def get_g_store(n, s_lin, s_quad):
    g_store = np.zeros(n)
    for x_val in range(n):
        g_store[x_val] = s_lin * x_val + s_quad * (x_val ** 2)
    return g_store


def get_g_rev(n, m, p, p_rev):
    g_rev = np.zeros([n, m, p])
    for x_val in range(n):
        for u_val in range(m):
            for w_val in range(p):
                g_rev[x_val, u_val, w_val] = -p_rev * min(x_val + u_val, w_val)
    return g_rev


def get_g_unmet(n, m, p, p_unmet):
    g_unmet = np.zeros([n, m, p])
    for x_val in range(n):
        for u_val in range(m):
            for w_val in range(p):
                g_unmet[x_val, u_val, w_val] = p_unmet * max(0, -x_val - u_val + w_val)
    return g_unmet


def get_g_sal(n, p_sal):
    g_sal = np.zeros(n)
    for x_val in range(n - 1):
        g_sal[x_val] = -p_sal * x_val
    g_sal[-1] = 10000
    return g_sal


def get_g_total(g_order, g_store, g_rev, g_unmet):
    g_order_w = np.repeat(g_order, g_rev.shape[2]).reshape(g_rev.shape)
    g_store_expanded = np.repeat(g_store, g_order.shape[1]).reshape(g_order.shape)
    g_store_expanded_w = np.repeat(g_store_expanded, g_rev.shape[2]).reshape(g_rev.shape)
    return g_order_w + g_store_expanded_w + g_rev + g_unmet


def get_transition_matrices(pol, d_t_dist):
    tms = []
    to_subtract = len(d_t_dist) - 1
    for time in range(pol.shape[1]):
        P = np.zeros([pol.shape[0], pol.shape[0]])
        to_add = pol[:, time]
        for idx, row in enumerate(P):
            start_idx = to_add[idx] - to_subtract + idx
            if start_idx >= 0:
                row[start_idx: start_idx + len(d_t_dist)] = d_t_dist
            else:
                num_in_zero = -start_idx + 1
                new_d_t_dist = np.zeros(len(d_t_dist) - num_in_zero + 1)
                new_d_t_dist[0] = np.sum(d_t_dist[0:num_in_zero])
                new_d_t_dist[1:] = d_t_dist[num_in_zero:]
                row[0:len(new_d_t_dist)] = new_d_t_dist
        tms.append(P)
    return tms


def part_c(pol, g_order, g_store, g_rev, g_unmet, g_sal, q_0, d_t_dist):
    tms = get_transition_matrices(pol, d_t_dist)
    g_order[-1, :] = 0
    g_order_t = []
    g_store_t = []
    g_rev_t = []
    g_unmet_t = []
    g_sal_t = []
    state = np.zeros(g_order.shape[0])
    state[q_0] = 1.
    for time in range(len(tms)):
        order_cost = 0
        rev_cost = 0
        unmet_cost = 0
        amt_ordered = pol[:, time]
        rev_mat = np.dot(g_rev, d_t_dist)
        unmet_mat = np.dot(g_unmet, d_t_dist)
        for idx, amt in enumerate(amt_ordered):
            order_cost += state[idx] * g_order[idx, amt]
            rev_cost += state[idx] * rev_mat[idx, amt]
            unmet_cost += state[idx] * unmet_mat[idx, amt]
        g_order_t.append(order_cost)
        g_rev_t.append(rev_cost)
        g_unmet_t.append(unmet_cost)
        g_store_t.append(np.dot(g_store, state))
        g_sal_t.append(0)
        state = np.dot(state, tms[time])
    g_sal_t[-1] = np.dot(g_sal, state)
    df = pd.DataFrame({'order_cost': g_order_t, 'store_cost': g_store_t, 'rev': g_rev_t, 'unmet_cost': g_unmet_t, 'sal': g_sal_t})
    ax = df.plot(title='5.1c: Costs broken down by type for Optimal Policy')
    ax.set_xlabel('time')
    ax.set_ylabel('cost')
    plt.ion()


def prob_1():
    d_t_dist, p_fixed, p_whole, p_disc, u_disc, s_lin, s_quad, p_rev, p_unmet, p_sal, T, C, D, q_0 = problem_data.hw5_p1_data()
    n = C + 2  # 0, 1, ..., C plus a dummy state
    m = C + 1  # 0, 1, ..., C
    p = len(d_t_dist)
    f = get_f(n, m, p)
    g_order = get_g_order(n, m, p_fixed, p_whole, p_disc, u_disc)  # n by m
    g_store = get_g_store(n, s_lin, s_quad)  # n
    g_rev = get_g_rev(n, m, p, p_rev)  # n by m by p
    g_unmet = get_g_unmet(n, m, p, p_unmet)
    g_sal = get_g_sal(n, p_sal)
    g_total = get_g_total(g_order, g_store, g_rev, g_unmet)
    pol, v = utils_mdp.value(f, g_total, g_sal, d_t_dist, T, g_is_w_dependent=True)
    utils.label('5.1a')
    print 'j_star: ', v[q_0, 0]
    # part b; yes the optimal policy converges; for all but the last two time periods it is optimal to buy 7 when x = 0,
    #  buy 5 when x = 1, and buy 0 otherwise
    ax = pd.DataFrame(pol).T.plot(title='5.1b: Optimal Policy for various current inventory values (most = 0)', legend=True)
    ax.set_xlabel('time')
    ax.set_ylabel('Number of items to buy')
    plt.ion()
    part_c(pol, g_order, g_store, g_rev, g_unmet, g_sal, q_0, d_t_dist)


if __name__ == '__main__':
    plt.show()
    prob_1()
