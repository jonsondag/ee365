import numpy as np
import pandas as pd
import problem_data
import random
import scipy.stats
import utils_io
import utils_mdp


def get_underlying_normal_mu(mu, var):
    return np.log((mu ** 2) / np.sqrt(mu ** 2 + var))


def get_underlying_normal_var(mu, var):
    return np.log(1 + var / (mu ** 2))


def get_f(n, m, p):
    f = np.zeros([n, m, p])
    for x_val in range(n):
        for u_val in range(m):
            for w_val in range(p):
                if u_val == 1 and x_val > 0:
                    f[x_val, u_val, w_val] = x_val - 1
                else:
                    f[x_val, u_val, w_val] = x_val
    return f.astype(int)


def get_g(n, m, T, p_mu, e_c):
    g = np.zeros([n, m, T])
    for x_val in range(n):
        for u_val in range(m):
            for time in range(T):
                if u_val == 1:
                    g[x_val, u_val, time] = p_mu[time] * e_c[x_val]
    return g


def get_g_w_known(n, m, T, price_grid, p_mu, e_c):
    price_grid_dim = price_grid.shape[0]
    g = np.zeros([n, m, price_grid_dim, T])
    for x_val in range(n):
        for u_val in range(m):
            for price_grid_idx in range(price_grid_dim):
                for time in range(T):
                    if u_val == 1:
                        g[x_val, u_val, price_grid_idx, time] = price_grid[price_grid_idx, time] * e_c[x_val]
    return g.reshape([g.shape[0], g.shape[1], g.shape[2], 1, g.shape[3]])


def get_v_final(n):
    v_final = np.inf * np.ones(n)
    v_final[0] = 0.
    return v_final


def add_g_w_known_to_values(g_time, f_w, next_values, len_x_in_n, m):
    next_values_mat = np.repeat(next_values, m).reshape(f_w.shape)
    idx_one = np.tile(np.arange(m), len_x_in_n).reshape(len_x_in_n, m)
    v_eval_at_f = next_values_mat[f_w, idx_one]
    g_w_known_plus_values = g_time + v_eval_at_f[..., np.newaxis]
    return g_w_known_plus_values


def print_cost(v, policy_name):
    print 'optimal expected cost for ' + policy_name + ':', str(v[-1, 0])


def get_schedule(pol):
    job_times = []
    min_job_time = 0
    for job in range(1, pol.shape[0])[::-1]:
        min_job_time = np.min(np.where(pol[job, min_job_time:] > 0)) + min_job_time
        job_times.append(min_job_time)
        min_job_time += 1
    print 'job times:', job_times
    return job_times


def plot_cost_histogram(job_times, p_mu_normal, p_var_normal, e_c):
    num_samples = 10000
    costs = np.zeros([num_samples, len(job_times)])
    p_mu_to_use = get_underlying_normal_mu(p_mu_normal, p_var_normal)
    p_var_to_use = get_underlying_normal_var(p_mu_normal, p_var_normal)
    p_std_to_use = np.sqrt(p_var_to_use)
    e_c_to_use = e_c[::-1][:-1]
    for idx, job_time in enumerate(job_times):
        costs[:, idx] = np.random.lognormal(p_mu_to_use[job_time], p_std_to_use[job_time], num_samples) * e_c_to_use[idx]
    sample_values = pd.Series(costs.sum(axis=1))
    utils_io.plot_histogram(sample_values, '5.3a: histogram of cost probabilities', 'Cost', 'Probability of Cost')


def plot_cost_histogram_real_time(pol, price_grid, e_c):
    num_samples = 10000
    num_jobs_to_run = pol.shape[0] - 1
    num_prices = pol.shape[1]
    num_time_steps = pol.shape[2]
    costs = []
    for sample in range(num_samples):
        sample_jobs_remaining = num_jobs_to_run
        this_cost = 0.
        for time_step in range(num_time_steps):
            random_price_idx = random.randint(0, num_prices - 1)
            do_sell = pol[sample_jobs_remaining, random_price_idx, time_step] == 1
            if do_sell and sample_jobs_remaining > 0:
                this_cost += price_grid[random_price_idx, time_step] * e_c[num_jobs_to_run - sample_jobs_remaining]
                sample_jobs_remaining -= 1
        costs.append(this_cost)
    utils_io.plot_histogram(pd.Series(costs), '5.3b: histogram of cost probabilities', 'Cost', 'Probability of Cost')


def get_prices_over_grid(price_grid_dim, p_mu_normal, p_var_normal):
    time_steps = len(p_mu_normal)
    price_grid = np.zeros([price_grid_dim, time_steps])
    sample_percentiles = (np.arange(price_grid_dim) + 0.5) / price_grid_dim
    for time_step in range(time_steps):
        dist = scipy.stats.lognorm([np.sqrt(p_var_normal[time_step])], loc=p_mu_normal[time_step])
        price_grid[:, time_step] = dist.ppf(sample_percentiles)  # ppf is the inverse of the cdf
    return price_grid


def prob_3():
    T, C, e_c, t, p_mu, p_var = problem_data.hw5_p3_data()
    n = C + 1
    m = 2
    p = 1
    f = get_f(n, m, p)
    g = get_g(n, m, T, p_mu, e_c)
    v_final = get_v_final(n)
    w_dist = np.array([1])

    # part a
    pol_a, v_a = utils_mdp.value(f, g, v_final, w_dist, T, g_is_time_dependent=True)
    utils_io.label('5.3a')
    print_cost(v_a, 'part a')
    job_times = get_schedule(pol_a)
    p_mu_normal = get_underlying_normal_mu(p_mu, p_var)
    p_var_normal = get_underlying_normal_var(p_mu, p_var)
    plot_cost_histogram(job_times, p_mu, p_var, e_c)

    # part b
    price_grid_dim = 10
    price_grid = get_prices_over_grid(price_grid_dim, p_mu_normal, p_var_normal)
    g_w_known = get_g_w_known(n, m, T, price_grid, p_mu, e_c)
    price_grid_dist = np.ones(len(price_grid)) / len(price_grid)
    f_w_known = f.repeat(price_grid_dim).reshape([n, m, price_grid_dim, 1])
    pol_b, v_b = utils_mdp.value_info_pat(f_w_known, g_w_known, v_final, price_grid_dist, w_dist, T, g_is_time_dependent=True)
    utils_io.label('5.3b')
    print_cost(v_b, 'part b')
    plot_cost_histogram_real_time(pol_b, price_grid, e_c)


if __name__ == '__main__':
    prob_3()
