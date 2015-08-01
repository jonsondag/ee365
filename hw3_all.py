import problem_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import scipy.stats
import seaborn
import utils


###########
# problem 1
###########

def get_transition_matrix():
    max_jobs = 26
    P = np.zeros([max_jobs, max_jobs])
    poisson_sample = scipy.stats.poisson.pmf(range(27), 2)
    for row in range(16):
        if row == 0:
            P[row, :] = poisson_sample[1:]
            P[row, 0] += poisson_sample[0]
        else:
            P[row, (row-1):] = poisson_sample[:-row]
            P[row, -1] += 1 - np.sum(poisson_sample[:-row])
    for row in range(16, 23):
        P[row, (row-2):] = poisson_sample[:-(row-1)]
        P[row, -1] += 1 - np.sum(poisson_sample[:-(row-1)])
    for row in range(23, max_jobs):
        P[row, (row-3):] = poisson_sample[:-(row-2)]
        P[row, -1] += 1 - np.sum(poisson_sample[:-(row-2)])
    return P


def get_profit(old_state, new_state):
    if old_state < 16:
        if old_state == 0 and new_state == 0:
            poisson_eq_zero = scipy.stats.poisson.pmf(0, 2)
            poisson_eq_one = scipy.stats.poisson.pmf(1, 2)
            if random.random() > poisson_eq_zero / (poisson_eq_zero + poisson_eq_one):  # at least 1 order arrived
                return 1.00
            else:
                return 0.00
        else:
            return 1.00
    elif 16 <= old_state < 23:
        return 0.50
    else:
        return 0.25


def get_profit_vec(subtract_ten=False):
    profit_vec = np.zeros(26)
    prob_0 = scipy.stats.poisson.pmf(0, 2)
    profit_vec[0] = prob_0 * -1 + (1 - prob_0) * 1
    profit_vec[1:16] = 1.0
    profit_vec[16:23] = 0.5
    profit_vec[23:] = 0.25
    poisson_size = 200
    poisson_sample = scipy.stats.poisson.pmf(range(poisson_size), 2)
    if subtract_ten:
        for val in range(26):
            if val <= 15:
                profit_vec[val] -= 10 * np.dot(range(poisson_size + val - 27), poisson_sample[(27 - val):])
            elif 16 <= val < 23:
                profit_vec[val] -= 10 * np.dot(range(poisson_size + val - 28), poisson_sample[(28 - val):])
            elif 23 <= val:
                profit_vec[val] -= 10 * np.dot(range(poisson_size + val - 29), poisson_sample[(29 - val):])
    return profit_vec


def part_c():
    df = pd.DataFrame(columns=['x', 'w'], index=range(101))
    df['x'][0] = 0
    for T in range(1, 101):
        last_x = df['x'][T-1]
        w = np.random.poisson(2)
        df['w'][T-1] = w
        if last_x < 16:
            num_processed = 1
        elif last_x < 23:
            num_processed = 2
        else:
            num_processed = 3
        next_x = last_x - num_processed + w
        if next_x > 25:
            next_x = 25
        if next_x < 0:
            next_x = 0
        df['x'][T] = next_x
    ax = df.plot(title='Sample system trajectory')
    ax.set_xlabel('time')
    ax.set_ylabel('value of x, w')
    plt.show()


def part_d(P):
    T = 101
    x = np.array([1.] + 25 * [0])
    for time in range(T + 1000):
        x = np.dot(x, P)
    x_ss = x
    profit_vec = get_profit_vec()

    v_t = x_ss * profit_vec
    for time in range(T):
        v_t = x_ss * profit_vec + np.dot(P, v_t)

    v_1 = v_t
    v_0 = x_ss * profit_vec + np.dot(P, v_1)
    utils.label('3.1d')
    print 'alpha: ', str((v_0 - v_1)[0])
    ax = pd.Series(v_0).plot(title='Value Function')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$v_0(x)$')
    plt.show()


def part_e(P):
    x = np.array([1.] + 25 * [0])
    for T in range(100):
        x = np.dot(x, P)
    utils.label('3.1e')
    print 'pct of time in slow mode: ', str(np.sum(x[:16]))
    print 'pct of time in normal mode: ', str(np.sum(x[16:23]))
    print 'pct of time in fast mode: ', str(np.sum(x[23:]))


def part_f(P):
    x = np.array([1.] + 25 * [0])
    for T in range(100):
        x = np.dot(x, P)
    utils.label('3.1f')
    print 'mean profit per step, large T: ', str(np.dot(x, get_profit_vec()))


def part_g(P):
    x = np.array([1.] + 25 * [0])
    for T in range(100):
        x = np.dot(x, P)
    utils.label('3.1g')
    print 'mean profit per step, large T with $10 penalty: ', str(np.dot(x, get_profit_vec(True)))


def prob_1():
    P = get_transition_matrix()
    part_c()
    part_d(P)
    part_e(P)
    part_f(P)
    part_g(P)

###########
# problem 2
###########

def get_p_modified(P, row):
    result = P.copy()
    result[row] = 0.
    result[row][row] = 1.
    return result


def part_2b(P):
    Q = get_p_modified(P, 0)
    state = np.zeros(6)
    state[1] = 1.
    time_horizon = 50
    vals = []

    for val in range(time_horizon):
        vals.append(state[0])
        state = np.dot(state, Q)
    ser = pd.Series(vals).diff()
    ser.fillna(0)
    ax = ser.plot(label='first passage time', title='3.2b, 3.2c Distribution of Passage Times Given Initial State i = 2', legend=True)
    ax.set_xlabel('passage time')
    ax.set_ylabel('probability')

    return ser


def part_2c(P):
    P_mod = np.zeros([P.shape[0] * 2, P.shape[1] * 2])
    P_mod[:P.shape[0], :P.shape[1]] = P
    P_mod[P.shape[0]:, P.shape[1]:] = P
    P_mod[0] = 0.
    P_mod[0, 6] = 0.4
    P_mod[0, 7] = 0.3
    P_mod[0, 9] = 0.3
    P_mod[6] = 0.
    P_mod[6, 6] = 1.
    time_horizon = 50
    vals = []
    state = np.zeros(12)
    state[1] = 1.

    for val in range(time_horizon):
        vals.append(state[6])
        state = np.dot(state, P_mod)
    ser = pd.Series(vals).diff()
    ser = ser.fillna(0)
    ser.plot(label='second passage time', legend=True)

    return ser


def part_2d(P, first_plus_second):
    state = np.zeros(6)
    state[1] = 1.
    time_horizon = 50
    vals = []
    for val in range(time_horizon):
        vals.append(state[0])
        state = np.dot(state, P)
    prob_in_state = pd.Series(vals)

    df = pd.DataFrame({'first or second passage time': first_plus_second, 'chance we are in state': prob_in_state})
    ax = df.plot(title='3.2d Distribution of first or second passage time, and chance we are in state for State i=2')
    ax.set_xlabel('time')
    ax.set_ylabel('probability')
    plt.ion()  # at first, hitting time probs directly trace the probability that we're in state 2,
    # then it tapers off exponentially (because we're likely to have already been in that state)


def part_2e(P, first_passage, second_passage):
    P_mod = np.zeros([P.shape[0] * 2, P.shape[1] * 2])
    P_mod[0, 6] = 0.4
    P_mod[0, 7] = 0.3
    P_mod[0, 9] = 0.3
    P_mod[P.shape[0]:, P.shape[1]:] = P
    state = np.zeros(12)
    state[0] = 1.
    P_mod[6] = 0.
    P_mod[6, 6] = 1.
    time_horizon = 50
    vals = []

    for val in range(time_horizon):
        vals.append(state[6])
        state = np.dot(state, P_mod)
    ser = pd.Series(vals).diff()
    recur_time = ser.fillna(0)

    df = pd.DataFrame({'second': second_passage, 'first': first_passage, 'recur_time': recur_time})
    ax = df.plot(title='3.2e Distribution of recurrence time (also including first and second passage times for clarity')
    ax.set_xlabel('time')
    ax.set_ylabel('probability')
    plt.ion()


def prob_2():
    P = problem_data.hw3_p2_data()
    first_passage = part_2b(P)
    second_passage = part_2c(P)
    first_plus_second = first_passage + second_passage
    part_2d(P, first_plus_second)
    part_2e(P, first_passage, second_passage)

###########
# problem 3
###########

def prob_3():
    n, P = problem_data.hw3_p3_data()
    R = reachable_states(P)
    C = communication_matrix(R)
    t = transience_vector(C, R)
    C_no_dupes = remove_duplicate_rows(C)
    num_transient_classes, C_no_dupes_sorted = move_transient_rows_up(C_no_dupes, t)
    C_adjacency = get_C_adjacency(R, C_no_dupes_sorted)
    L = topological_sort(C_adjacency[:num_transient_classes, :num_transient_classes])
    L = np.concatenate((L, range(len(L), C_no_dupes.shape[0])))
    C_ordered = C_no_dupes[L]
    P_index_order = get_index_order(C_ordered)
    P_ordered = P[P_index_order]
    utils.label('3.3')
    print 'Transition matrix formatted for class decomposition:'
    print P_ordered

def reachable_states(A):
    n = A.shape[0]
    result = np.where(A > 0, 1., 0.)
    while True:
        orig_result_this_pass = np.copy(result)
        for i in range(n):
            for k in range(n):
                if result[i, k] == 1:
                    for j in range(n):
                        if result[k, j] == 1:
                            result[i, j] = 1
        if np.array_equal(orig_result_this_pass, result):
            break
    return result


def communication_matrix(R):
    R_t = R.T
    return np.where((R > 0) & (R_t > 0), 1., 0.)


def transience_vector(C, R):
    with_outgoing = (R - C).sum(axis=1)
    return np.where(with_outgoing, 1., 0.)


def topological_sort(A):
    binary = np.where(A > 0, 1., 0.)
    L = []
    S = get_nodes_with_no_incoming_edges(binary)
    while len(S) > 0:
        val = S.pop()
        L.append(val)
        binary[:, val] = 0.
        binary[val, :] = 0.
        S = S.union(get_nodes_with_no_incoming_edges(binary))
    return np.array(L)


def remove_duplicate_rows(C):
    C = np.ascontiguousarray(C)
    unique_C = np.unique(C.view([('', C.dtype)]*C.shape[1]))
    return unique_C.view(C.dtype).reshape((unique_C.shape[0], C.shape[1]))


def move_transient_rows_up(C_no_dupes, t):
    num_transients_in_class = np.logical_and(C_no_dupes, t).sum(axis=1)
    num_transient_classes = np.sum(np.where(num_transients_in_class > 0, 1., 0.))
    return num_transient_classes, C_no_dupes[np.argsort(num_transients_in_class)[::-1]]


def get_nodes_with_no_incoming_edges(binary):
    return set(np.unique(np.where(binary.sum(axis=0) == 1)[0]))


def get_C_adjacency(R, C_no_dupes_sorted):
    reachability = np.dot(np.dot(C_no_dupes_sorted, R), C_no_dupes_sorted.T)
    return np.where(reachability > 0, 1., 0.)


def get_index_order(C_ordered):
    result = np.array([])
    for row in C_ordered:
        row_idx_gt_0 = np.where(row > 0, range(len(row)), 0)
        result = np.concatenate((result, row_idx_gt_0[row_idx_gt_0 > 0]))
    return result.tolist()


def test_topological_sort():
    A = np.array([[1., 0., 0., 0., 0.],
                  [1., 1., 0., 0., 0.],
                  [1., 1., 1., 0., 0.],
                  [0., 1., 0., 1., 0.],
                  [1., 1., 1., 1., 1.]])
    topological_sort(A)

###########
# problem 4
###########

def prob_4():
    n, L, R, theta = get_starting_values()
    trans_mat = get_trans_mat(L, theta)
    part_a(trans_mat)
    part_b_mc(trans_mat, R)
    part_b_dist_prop(trans_mat, R)
    part_b_value_iter(trans_mat, R)


def get_starting_values():
    n = 100
    L = np.zeros([n, n])
    num_links = np.concatenate((np.zeros(20),
                                np.floor(np.random.random(75) * 3 + 1),
                                np.floor(np.random.random(5) * 6 + 10)))
    for i in range(n):
        links = np.floor(np.random.random(num_links[i]) * n)  # dimension i, random numbers in 1 to num_links
        L[links.tolist(), i] = 1
        L[i, i] = 0
    to_zero = np.floor(np.random.random(2) * n)
    L[to_zero.tolist(), :] = 0
    R = np.random.random([n, n]) * L
    theta = 1 - 1e-2
    return n, L, R, theta


def get_trans_mat(L, theta):
    n = L.shape[0]
    L_trans = L.copy()
    L_trans[np.where(L_trans.sum(axis=1) == 0), :] = float(n - 1) / n
    for val in range(n):
        L_trans[val, val] = 0
    o_mat = np.repeat(L_trans.sum(axis=1), n).reshape([n, n])
    has_link = theta / o_mat
    not_has_link = (1 - theta) / ((n - 1) * np.ones([100, 100]) - o_mat)
    trans_mat = np.where(L_trans > 0, has_link, not_has_link)
    for val in range(n):
        trans_mat[val, val] = 0
    return trans_mat


def part_a(trans_mat):
    n = trans_mat.shape[0]
    dist = float(1) / n * np.ones(n)
    utils.label('3.4a')
    for t in range(101):
        dist = np.dot(dist, trans_mat)
        if t == 10 or t == 100:
            print 'Surfer\'s most likely page at time %d is %d' % (t, np.argmax(dist))


def part_b_mc(trans_mat, R):
    num_samples = 1000
    values = []
    for sample in range(num_samples):
        value = 0
        state = np.floor(random.random() * 100)
        for t in range(50):
            next_state_dist = trans_mat[state, :]
            next_state = np.min(np.where(random.random() < next_state_dist.cumsum()))
            value += R[state, next_state]
            state = next_state
        values.append(value)
    utils.label('3.4b')
    print 'J is expected total payment...'
    print 'MC estimate of J, t=0,...,50:', str(np.mean(values))


def part_b_dist_prop(trans_mat, R):
    values_by_state = get_values_by_state(trans_mat, R)
    total_value = 0
    state = np.ones(100) / 100
    for t in range(50):
        total_value += np.dot(values_by_state, state)
        state = np.dot(state, trans_mat)
    print 'Distribution propogation estimate of J, t=0,...,50:', str(total_value)


def part_b_value_iter(trans_mat, R):
    values_by_state = get_values_by_state(trans_mat, R)
    initial_state = np.ones(100) / 100
    values = np.zeros(100)
    for t in range(50):
        values = values_by_state + np.dot(trans_mat, values)
    total_value = np.dot(initial_state, values)
    print 'Value iteration estimate of J, t=0,...,50:', str(total_value)


def get_values_by_state(trans_mat, R):
    return np.sum(trans_mat * R, axis=1)


if __name__ == '__main__':
    plt.show()
    prob_1()
    prob_2()
    # test_topological_sort()
    prob_3()
    prob_4()

