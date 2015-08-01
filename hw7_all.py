import numpy as np
import os
import pandas as pd
import problem_data
import utils_fns
import utils


###########
# problem 1
###########
def p_1():
    A, b, x, _, _, _, _, _, _ = problem_data.hw7_p1_data()
    lin_func = utils_fns.LinearFunction(A, b)
    first_result = np.linalg.norm(lin_func.evaluate(x), ord=2) ** 2
    utils.label('7.1')
    print 'norm squared direct calc result: ', str(first_result)
    second_result = lin_func.norm_squared_linear().evaluate(x)
    print 'norm squared calc via quadratic function: ', str(second_result)


###########
# problem 2
###########
def p_2b():
    _, _, x, P, q, r, y, _, _ = problem_data.hw7_p1_data()
    quad_func = utils_fns.QuadraticFunction(P, q, r)
    quad_func_partial = quad_func.partial_evaluation(y, len(y))
    first_result = quad_func_partial.evaluate(x)
    utils.label('7.2b')
    print 'quadratic function partial evaluation result: ', str(first_result)
    second_result = quad_func.evaluate(np.concatenate((x, y)))
    print 'quadratic function direct evaluation result: ', str(second_result)


def p_2e():
    _, _, x, P, q, r, _, y_vals, y_pmf = problem_data.hw7_p1_data()
    y_mean = np.dot(y_vals.T, y_pmf).T[0]
    devs = y_vals - y_mean
    y_cov = np.dot(y_pmf.T * devs.T, devs)
    quad_func = utils_fns.QuadraticFunction(P, q, r)
    h_x = quad_func.partial_expectation(y_mean, y_cov)
    first_result = h_x.evaluate(x)
    utils.label('7.2e')
    print 'partial expectation via quadratic function: ', str(first_result)
    second_result = 0.
    for idx, y_val in enumerate(y_vals):
        eval_point = np.vstack((x, y_val[:, np.newaxis]))
        second_result += y_pmf[idx] * quad_func.evaluate(eval_point)
    print 'partial expectation via direct evaluation: ', str(second_result[0])


###########
# problem 3
###########
def ss_lqsc(f, g, w_bar, w_var, n, m, constraints_A=None, constraints_b=None):
    last_value_func = utils_fns.QuadraticFunction(np.zeros([n, n]), np.zeros([n, 1]), 0)
    max_time_steps = 500
    epsilon = 1e-6
    r_change = np.inf
    lfu = None

    for time in range(max_time_steps):
        next_stage_value_func = last_value_func.precompose_linear(f)
        next_stage_value_func_exp = next_stage_value_func.partial_expectation(w_bar, w_var)
        overall_value_func = next_stage_value_func_exp.plus_quadratic(g)
        if constraints_A is None or constraints_b is None:
            qfx, lfu = overall_value_func.partial_minimization(m)
        else:
            qfx, lfu = overall_value_func.partial_minimization_with_constraint(m, constraints_A, constraints_b)
        if qfx.difference(last_value_func, include_r=False) < epsilon:
            r_change = qfx.r - last_value_func.r
            last_value_func = qfx
            break
        last_value_func = qfx
    return last_value_func, lfu, r_change


def p_3():
    A, B, P, q, r, w_bar, w_var, n, m = problem_data.hw7_p3_data()
    g = utils_fns.QuadraticFunction(P, q, r)
    A_f = np.hstack((A, B, np.eye(n)))
    b_f = np.zeros([A_f.shape[0], 1])
    f = utils_fns.LinearFunction(A_f, b_f)
    val, pol, _ = ss_lqsc(f, g, w_bar, w_var, n, m)
    utils.label('7.3')
    print 'optimal steady-state controller values:'
    print pol
    print 'optimal steady-state value function (disregard r-value as we are only interested in quadratic and linear components):'
    print val


###########
# problem 4
###########
class P4Params(object):
    def __init__(self, m, n, T, B, Cb, Cr, Cd, lambda_, clip):
        self.m = m
        self.n = n
        self.T = T
        self.B = B
        self.Cb = Cb
        self.Cr = Cr
        self.Cd = Cd
        self.lambda_ = lambda_
        self.clip = clip


def get_u_based_on_descending_vals(x, params):
    x_mod = x.copy().reshape([params.m, params.n])
    u = np.zeros(x_mod.shape)

    while x_mod.max() > 0:
        next_idx = np.unravel_index(x_mod.argmax(), x_mod.shape)
        u[next_idx] = 1.
        x_mod[next_idx[0], :] = 0  # no more packets sent from this input
        x_mod[:, next_idx[1]] = 0  # no more packets sent to this output
    return u.reshape([params.m * params.n, 1])


def strategy_4b(x, params):
    return get_u_based_on_descending_vals(x, params)


def get_w(lambda_):
    return np.random.poisson(lambda_)


def drop_overflow_packets(x, Cd, B, cost):
    x_new = np.minimum(x, B)
    cost = cost + Cd * (x - x_new).sum()
    return x_new, cost


def calculate_buffer_cost(x, Cb, cost):
    return cost + Cb * x.sum()


def calculate_routing_reward(u, Cr, cost):
    return cost - Cr * u.sum()


def simulate_strategy(strategy, params):
    x = np.zeros([params.m * params.n, 1])
    cost = 0.
    for _ in range(params.T):
        w = get_w(params.lambda_)
        x = x + w
        x, cost = drop_overflow_packets(x, params.Cd, params.B, cost)
        u = strategy(x, params)
        cost = calculate_routing_reward(u, params.Cr, cost)
        x = x - u
        cost = calculate_buffer_cost(x, params.Cb, cost)
    return cost / params.T


def get_g_fn(params, rho_1, rho_2, rho_3):
    mn = params.m * params.n
    m = params.m
    n = params.n

    num_x_and_u_vars = mn + mn + m + n  # x, u, u input sums, u output sums
    P_g = np.zeros([num_x_and_u_vars, num_x_and_u_vars])
    P_g[:mn, :mn] = np.diag(params.lambda_.ravel())
    P_g[mn:2 * mn, mn:2 * mn] = (rho_1 / mn) * np.eye(mn)
    P_g[2 * mn:2 * mn + m, 2 * mn:2 * mn + m] = (rho_2 / m) * np.eye(m)
    P_g[2 * mn + m:2 * mn + m + n, 2 * mn + m:2 * mn + m + n] = (rho_3 / n) * np.eye(n)

    q_g = np.zeros([num_x_and_u_vars, 1])
    r_g = 0

    # account for -1/2 in u_ij term
    q_g[mn:2 * mn, 0] = -1 * (rho_1 / (2 * mn))
    r_g += 0.25 * mn * (rho_1 / (2 * mn))
    # account for -1 in input sum terms
    q_g[2 * mn:2 * mn + m, 0] = -2 * (rho_2 / (2 * m))
    r_g += 1.0 * m * (rho_2 / (2 * m))
    # account for -1 in output sum terms
    q_g[2 * mn + m:2 * mn + m + n, 0] = -2 * (rho_3 / (2 * n))
    r_g += 1.0 * n * (rho_3 / (2 * n))

    return utils_fns.QuadraticFunction(P_g, q_g, r_g)


def get_constraints(params):
    mn = params.m * params.n
    m = params.m
    n = params.n

    constraints_A = np.zeros([m + n, mn + m + n])
    constraints_b = np.zeros([m + n, 1])

    # sum across inputs equals input sum
    for m_val in range(m):
        constraints_A[m_val, n * m_val:n * (m_val + 1)] = 1.0
        constraints_A[m_val, mn + m_val] = -1.0

    # sum across outputs equals output sum
    for n_val in range(n):
        for m_val in range(m):
            constraints_A[m + n_val, m_val * n + n_val] = 1.0
            constraints_A[m + n_val, mn + m + n_val] = -1.0

    return constraints_A, constraints_b


def strategy_4d(x, lfu, params):
    u_size = params.m * params.n

    u = lfu.evaluate(x)
    u = u[:u_size]

    # rounding strategy
    u = np.maximum(0, u)
    u = np.minimum(x, u)
    u = get_u_based_on_descending_vals(u, params)

    return u


def get_strategy_4d(lfu):
    return lambda x_val, params: strategy_4d(x_val, lfu, params)


def strategy_4e(x, value_func, params):
    w_avg = np.zeros([params.m * params.n, 1])
    num_adp_samples = 100
    for sample in range(num_adp_samples):
        w = get_w(params.lambda_)
        w_avg += w
    w_avg /= num_adp_samples
    # set d/du = 0; d/du = 2*P*u - 2*x - 2*(1/N)*sum(w) - 2*q
    u = np.dot(np.linalg.inv(value_func.P), x + w_avg + value_func.q)
    u = np.maximum(0, u)
    u = np.minimum(x, u)
    u = get_u_based_on_descending_vals(u, params)

    return u


def get_strategy_4e(value_func):
    return lambda x_val, params: strategy_4e(x_val, value_func, params)


def p4_d_e(params):
    mn = params.m * params.n
    Ax = np.eye(mn)
    Au = np.hstack([-np.eye(mn), np.zeros([mn, params.m + params.n])])
    Aw = np.eye(mn)
    A_f = np.hstack([Ax, Au, Aw])
    b_f = np.zeros(mn)
    f = utils_fns.LinearFunction(A_f, b_f)
    rho_1_vals = np.linspace(0.2, 1, 5)
    rho_2_vals = np.linspace(0.2, 1, 5)
    rho_3_vals = np.linspace(0.2, 1, 5)
    df = pd.DataFrame(columns=['strategy_name', 'rho_1', 'rho_2', 'rho_3', 'avg_cost_ss', 'avg_cpst_adp'])

    for rho_1 in rho_1_vals:
        for rho_2 in rho_2_vals:
            for rho_3 in rho_3_vals:
                g = get_g_fn(params, rho_1, rho_2, rho_3)
                constraints_A, constraints_b = get_constraints(params)
                value_func, lfu, r_change = ss_lqsc(f, g, params.lambda_.ravel(), np.diag(params.lambda_.ravel()), mn,
                                                    mn + params.m + params.n, constraints_A, constraints_b)
                strategy_ss = get_strategy_4d(lfu)
                avg_cost_ss = simulate_strategy(strategy_ss, params)
                strategy_name = '7.4, ' + str(rho_1) + ',' + str(rho_2) + ',' + str(rho_3)
                strategy_adp = get_strategy_4e(value_func)
                avg_cost_adp = simulate_strategy(strategy_adp, params)
                df = df.append({'strategy_name': strategy_name, 'rho_1': rho_1, 'rho_2': rho_2, 'rho_3': rho_3,
                                'avg_cost_ss': avg_cost_ss, 'avg_cost_adp': avg_cost_adp}, ignore_index=True)
    # df.to_csv(os.path.expanduser('~') + '/crossbar_switch.csv')
    strat_ss = df.sort('avg_cost_ss').iloc[0]
    utils.label('7.4d')
    print 'best optimal steady-state strategy has avg_cost={0:.3f} with rho_1={1:.2f}, rho_2={2:.2f}, rho_3={3:.2f}'. \
        format(strat_ss['avg_cost_ss'], strat_ss['rho_1'], strat_ss['rho_2'], strat_ss['rho_3'])
    strat_adp = df.sort('avg_cost_adp').iloc[0]
    utils.label('7.4e')
    print 'best optimal adp strategy has avg_cost={0:.3f} with rho_1={1:.2f}, rho_2={2:.2f}, rho_3={3:.2f}'. \
        format(strat_adp['avg_cost_adp'], strat_adp['rho_1'], strat_adp['rho_2'], strat_adp['rho_3'])


def p_4():
    p4_params = P4Params(*problem_data.hw7_p4_data())
    utils.label('7.4b')
    print 'average heuristic policy stage cost is: {0:.3f}'.format(simulate_strategy(strategy_4b, p4_params))
    p4_d_e(p4_params)


if __name__ == '__main__':
    p_1()
    p_2b()
    p_2e()
    p_3()
    p_4()
