import problem_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import utils_io

###########
# problem 4
###########

def get_all_costs(T, a, w_seq, pw, phi_cl, ct):
    all_costs = np.zeros(2**(2*2*(T-1)))
    all_cost_arrs = []
    min_prescient_costs = np.inf * np.ones(len(w_seq))
    min_prescient_cost_indices = np.zeros(len(w_seq))
    for k in range(2**(2*2*(T-1))):
        pc = phi_cl[k]
        this_cost = 0.0
        all_cost_arrs.append([])
        for idx, w in enumerate(w_seq):
            u = np.zeros(T)
            u[0] = pc[0, w[0], 0]
            u[1] = pc[u[0], w[1], 1]
            u[2] = pc[u[1], w[2], 2]
            u[3] = pc[u[2], w[3], 3]
            x = np.concatenate((np.zeros(1), u[:-1]))
            costs = np.select([(x == 0) & (u == 0) & (w == 0), (x == 0) & (u == 1)], [-a, ct], default=0)
            tot_cost = np.sum(costs)
            if tot_cost < min_prescient_costs[idx]:
                min_prescient_costs[idx] = tot_cost
                min_prescient_cost_indices[idx] = k
            prob = pw[idx]
            this_cost += tot_cost * prob
            all_cost_arrs[k].append(tot_cost)
        all_costs[k] = this_cost
    return all_costs, all_cost_arrs, min_prescient_costs


def prob_4():
    T, p, a, w_seq, pw, phi_cl, ct = problem_data.hw1_p4_data()
    all_costs, all_cost_arrs, min_prescient_costs = get_all_costs(T, a, w_seq, pw, phi_cl, ct)

    # 4a
    min_ol_cost = np.inf
    min_ol_idx = None
    min_cl_cost = np.inf
    min_cl_idx = None
    for idx, pc in enumerate(phi_cl):
        if np.array_equal(pc[0, :, :], pc[1, :, :]):
            is_open_loop = True
        else:
            is_open_loop = False
        if is_open_loop:
            if all_costs[idx] < min_ol_cost:
                min_ol_cost = all_costs[idx]
                min_ol_idx = idx
        if all_costs[idx] < min_cl_cost:
            min_cl_cost = all_costs[idx]
            min_cl_idx = idx
    utils_io.label('1.4a')
    print 'min_ol_cost: ', str(min_ol_cost)
    print 'min_ol_policy: ', phi_cl[min_ol_idx]
    # print 'min_ol_costs_by_w: ', all_cost_arrs[min_ol_idx]

    df = pd.DataFrame({'min_costs': all_cost_arrs[min_ol_idx], 'open loop costs': pw})
    df = df.groupby('min_costs').sum()

    # 4b
    utils_io.label('1.4b')
    print 'expected cost, optimal prescient policy: ', str(np.sum(min_prescient_costs * pw))
    # print 'min_prescient_costs_by_w: ', min_prescient_costs

    df_prescient = pd.DataFrame({'min_costs': min_prescient_costs, 'prescient costs': pw})
    df_prescient = df_prescient.groupby('min_costs').sum()
    df = df.join(df_prescient, how='outer')

    # 4c
    utils_io.label('1.4c')
    print 'min_cl_cost: ', str(min_cl_cost)
    print 'min_cl_policy: ', phi_cl[min_cl_idx]

    df_cl = pd.DataFrame({'min_costs': all_cost_arrs[min_cl_idx], 'closed loop costs': pw})
    df_cl = df_cl.groupby('min_costs').sum()
    df = df.join(df_cl, how='outer')
    df = df.join(pd.DataFrame(index=range(int(df.index.min()), int(df.index.max() + 1))), how='outer')
    ax = df.plot(kind='bar', sharex=True, sharey=True, title='1.4c Probability of costs by strategy')
    ax.set_xlabel('cost')
    ax.set_ylabel('probability')
    plt.ion()


if __name__ == '__main__':
    plt.show()
    prob_4()
